import os
import shutil
import time
import glob
import torch
from PIL import Image
import cv2
import random
import string
import numpy as np
import random
from datetime import datetime

SAVE_DIR = "./annotated_images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

yolo_image_mapping = {
    '1': 0, '2': 1, '3': 2,'4': 3,'5': 4,'6': 5,'7': 6,'8': 7,'9': 8,'A': 9,'B': 10,
    'BullsEye': 11,'C': 12,'D': 13,'Dot': 14,'Down': 15,'E': 16,'F': 17,'G': 18,'H': 19,
    'Left': 20,'Right': 21,'S': 22,'T': 23,'U': 24,'Up': 25,'V': 26,'W': 27,'X': 28,
    'Y': 29,'Z': 30
}

name_to_id = {
    "NA": 'NA',"BullsEye": 10,"1": 11,"2": 12,"3": 13,"4": 14,"5": 15,
    "6": 16,"7": 17,"8": 18,"9": 19,"A": 20,"B": 21,"C": 22,"D": 23,"E": 24,
    "F": 25,"G": 26,"H": 27,"S": 28,"T": 29,"U": 30,"V": 31,"W": 32,"X": 33,"Y": 34,
    "Z": 35,"Up": 36,"Down": 37,"Right": 38,"Left": 39,"Up Arrow": 36,"Down Arrow": 37,
    "Right Arrow": 38,"Left Arrow": 39,"Dot": 40
}

def get_random_string(length):
    """
    Generate a random string of fixed length 

    Inputs
    ------
    length: int - length of the string to be generated

    Returns
    -------
    str - random string

    """
    result_str = ''.join(random.choice(string.ascii_letters) for i in range(length))
    return result_str

def load_model():
    """
    Load the model from the local directory
    """
    model = torch.hub.load('./', 'custom', path='best.pt', source='local')
    return model


def display_image(frame, window_name):
    """
    Displays an image in a window that stays open indefinitely.
    Inputs
    ------
    frame: numpy array - the image to display
    window_name: str - the name of the display window
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, frame)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(0)  # The window will remain open indefinitely until a key is pressed
    cv2.destroyWindow(window_name)

def draw_own_bbox(img,x1,y1,x2,y2,label,color=(36,255,12),text_color=(0,0,0)):
    """
    Draw bounding box on the image with text label and save both the raw and annotated image in the 'own_results' folder

    Inputs
    ------
    img: numpy.ndarray - image on which the bounding box is to be drawn

    x1: int - x coordinate of the top left corner of the bounding box

    y1: int - y coordinate of the top left corner of the bounding box

    x2: int - x coordinate of the bottom right corner of the bounding box

    y2: int - y coordinate of the bottom right corner of the bounding box

    label: str - label to be written on the bounding box

    color: tuple - color of the bounding box

    text_color: tuple - color of the text label

    Returns
    -------
    img - image

    """
    # Reformat the label to {label name}-{label id}
    label = label + "-" + str(name_to_id[label])
    # Convert the coordinates to int
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    # Create a random string to be used as the suffix for the image name, just in case the same name is accidentally used
    rand = str(int(time.time()))

    # Save the raw image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"own_results/raw_image_{label}_{rand}.jpg", img)

    # Draw the bounding box
    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    # For the text background, find space required by the text so that we can put a background with that amount of width.
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    # Print the text  
    img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
    img = cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

    return img

def predict_image(image_bytes, obstacle_id, signal, model):
    """
    Process the image and return the best prediction based on the robot's signal.

    Inputs
    ------
    image_bytes: bytes - the image data in bytes
    obstacle_id: str - the obstacle ID
    signal: str - the direction signal ('L', 'R', 'C') to filter predictions
    model: torch.hub.load - model to be used for prediction

    Returns
    -------
    tuple - (final image ID, annotated image)
    """
    try:
        # Convert the bytes to a NumPy array and decode the image using OpenCV
        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return 'NA', None

        # Run object detection using the YOLO model
        results = model(frame)

        # Extract bounding boxes and other details directly from results
        pred_list = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bboxHt = y2 - y1
            bboxWt = x2 - x1
            bboxArea = bboxHt * bboxWt
            class_id = int(box.cls.item())
            confidence = box.conf.item()
            class_name = next((key for key, value in yolo_image_mapping.items() if value == class_id), "Unknown")

            if class_name != 'BullsEye':  # Filter out BullsEye
                pred_list.append({
                    'xmin': x1,
                    'ymin': y1,
                    'xmax': x2,
                    'ymax': y2,
                    'bboxArea': bboxArea,
                    'confidence': confidence,
                    'name': class_name
                })
            else: # If the image is Bullseye
                return obstacle_id, "10"

        # Sort the predictions by bbox area
        pred_list = sorted(pred_list, key=lambda x: x['bboxArea'], reverse=True)

        # Initialize prediction to NA
        pred = 'NA'

        # If only one prediction is detected, use it
        if len(pred_list) == 1:
            pred = pred_list[0]

        # If more than one label is detected, filter by confidence, area, and the signal (L, R, C)
        elif len(pred_list) > 1:
            pred_shortlist = []
            current_area = pred_list[0]['bboxArea']

            # Filter by confidence and area
            for row in pred_list:
                if row['confidence'] > 0.5 and ((current_area * 0.8 <= row['bboxArea']) or (row['name'] == 'One' and current_area * 0.6 <= row['bboxArea'])):
                    pred_shortlist.append(row)
                    current_area = row['bboxArea']

            if len(pred_shortlist) == 1:
                pred = pred_shortlist[0]
            else:
                # Filter further using the signal ('L', 'R', 'C')
                pred_shortlist.sort(key=lambda x: x['xmin'])

                if signal == 'L':
                    pred = pred_shortlist[0]  # Leftmost prediction
                elif signal == 'R':
                    pred = pred_shortlist[-1]  # Rightmost prediction
                elif signal == 'C':
                    for row in pred_shortlist:
                        if 250 < row['xmin'] < 774:  # Central prediction
                            pred = row
                            break
                    if isinstance(pred, str):  # If no central prediction found
                        pred = pred_shortlist[-1]  # Largest area

        # Annotate the image with the selected prediction
        if isinstance(pred, dict):
            x1, y1, x2, y2 = pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax']
            label = pred['name']
            image_id = str(name_to_id.get(label, 'NA'))

            # Draw the bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}, ID: {image_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Save the annotated frame
            save_path = os.path.join(SAVE_DIR,
                    f"{obstacle_id}_{image_id}_{datetime.now()}.jpg")
            cv2.imwrite(save_path, frame)
            display_image(frame, f"Obstacle {obstacle_id}, Class: {label}")

            return image_id, frame
        else:
            return 'NA', frame

    except Exception as e:
        print(f"Error during prediction: {e}")
        return 'NA', None


def predict_image_week_9(image, model):
    # Load the image
    img = Image.open(os.path.join('uploads', image))
    # Run inference
    results = model(img)
    # Save the results
    results.save('runs')
    # Convert the results to a dataframe
    df_results = results.pandas().xyxy[0]
    # Calculate the height and width of the bounding box and the area of the bounding box
    df_results['bboxHt'] = df_results['ymax'] - df_results['ymin']
    df_results['bboxWt'] = df_results['xmax'] - df_results['xmin']
    df_results['bboxArea'] = df_results['bboxHt'] * df_results['bboxWt']

    # Label with largest bbox height will be last
    df_results = df_results.sort_values('bboxArea', ascending=False)
    pred_list = df_results 
    pred = 'NA'
    # If prediction list is not empty
    if pred_list.size != 0:
        # Go through the predictions, and choose the first one with confidence > 0.5
        for _, row in pred_list.iterrows():
            if row['name'] != 'Bullseye' and row['confidence'] > 0.5:
                pred = row    
                break

        # Draw the bounding box on the image 
        if not isinstance(pred,str):
            draw_own_bbox(np.array(img), pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax'], pred['name'])
        
    # Dictionary is shorter as only two symbols, left and right are needed
    name_to_id = {
        "NA": 'NA',
        "Bullseye": 10,
        "Right": 38,
        "Left": 39,
        "Right Arrow": 38,
        "Left Arrow": 39,
    }
    # Return the image id
    if not isinstance(pred,str):
        image_id = str(name_to_id[pred['name']])
    else:
        image_id = 'NA'
    return image_id


def stitch_images(image_dir, save_stitched_path, min_obstacle_id=1, max_obstacle_id=8):
    """
    Stitch the latest images from a specified obstacle range.

    Inputs
    ------
    image_dir: str - the directory where images are stored
    save_stitched_path: str - the path to save the stitched image
    min_obstacle_id: int - minimum obstacle ID to consider
    max_obstacle_id: int - maximum obstacle ID to consider

    Returns
    -------
    str - path to the saved stitched image
    """
    images = []
    latest_images = {}

    for filename in os.listdir(image_dir):
        parts = filename.split('_')
        if len(parts) < 3:
            continue  # Skip files that don't match the expected format

        try:
            obstacle_id = int(parts[0])
            timestamp_str = parts[2].replace('.jpg', '')
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
        except (ValueError, IndexError):
            continue

        if min_obstacle_id <= obstacle_id <= max_obstacle_id:
            if obstacle_id not in latest_images or latest_images[obstacle_id]['timestamp'] < timestamp:
                latest_images[obstacle_id] = {'timestamp': timestamp, 'filename': filename}

    for obstacle_id in sorted(latest_images.keys()):
        img_path = os.path.join(image_dir, latest_images[obstacle_id]['filename'])
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)

    if len(images) < 2:
        return "Error: At least two images are required for stitching."

    stitched_image = cv2.hconcat(images)  # Horizontally concatenate the images

    cv2.imwrite(save_stitched_path, stitched_image)
    return stitched_image

def stitch_image_own():
    """
    Stitches the images in the folder together and saves it into own_results folder

    Basically similar to stitch_image() but with different folder names and slightly different drawing of bounding boxes and text
    """
    imgFolder = 'own_results'
    stitchedPath = os.path.join(imgFolder, f'stitched-{int(time.time())}.jpeg')

    imgPaths = glob.glob(os.path.join(imgFolder+"/annotated_image_*.jpg"))
    imgTimestamps = [imgPath.split("_")[-1][:-4] for imgPath in imgPaths]
    
    sortedByTimeStampImages = sorted(zip(imgPaths, imgTimestamps), key=lambda x: x[1])

    images = [Image.open(x[0]) for x in sortedByTimeStampImages]
    width, height = zip(*(i.size for i in images))
    total_width = sum(width)
    max_height = max(height)
    stitchedImg = Image.new('RGB', (total_width, max_height))
    x_offset = 0

    for im in images:
        stitchedImg.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    stitchedImg.save(stitchedPath)

    return stitchedImg

