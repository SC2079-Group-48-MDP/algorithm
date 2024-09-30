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

def draw_label(image, x1, y1, x2, y2, label_text):
    """
    Draws a label on the image based on bounding box coordinates.
    Parameters:
        image (np.array): The image where labels will be drawn.
        x1, y1, x2, y2 (int): Coordinates of the bounding box.
        label_text (str): Text to put as a label.
    Returns:
        np.array: The image with the label drawn.
    """
    img_height, img_width = image.shape[:2]

    # Calculate label position to avoid being cut off
    label_position = (x1, max(20, y1 - 10))  # Default above the box
    if y1 < 20:
        label_position = (x1, y2 + 20)  # Move below the box if too close to top

    # Ensure label does not go beyond image boundaries
    if x1 + 7 * len(label_text) > img_width:  # Estimate width of text
        label_position = (img_width - 7 * len(label_text), label_position[1])

    # Draw bounding box and label
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image


async def predict_image(image_bytes, obstacle_id, model):
    # Convert the bytes data to a NumPy array
    image_array = np.frombuffer(image_bytes, np.uint8)
    # Decode the image using OpenCV
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Check if the frame is received correctly
    if frame is not None:
        # Retrieve the dimensions of the frame
        img_height, img_width, _ = frame.shape

        # Run object detection using the provided model
        results = model(frame)

        # Check if any objects were detected
        if not results or len(results[0].boxes) == 0:
            return "NA", None

        # Find the largest bounding box
        max_area = 0
        selected_box = None
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                selected_box = box

        if selected_box is None:
            return "NA", None

        # Process the selected result
        x1, y1, x2, y2 = map(int, selected_box.xyxy[0])
        class_id = int(selected_box.cls.item())
        class_name = next((key for key, value in yolo_image_mapping.items() if value == class_id), "Unknown")
        final_image_id = name_to_id.get(class_name, "NA")

        # Adjust the class_name if it is 'Dot'
        adjusted_class_name = "Stop" if class_name == "Dot" else class_name

        # Prepare the label_text with the adjusted class name and the image ID
        label_text = f"{adjusted_class_name}, Image ID: {final_image_id}"

        # Draw label
        frame = draw_label(frame, x1, y1, x2, y2, label_text)

        # Save the annotated frame
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_path = os.path.join(SAVE_DIR, f"{obstacle_id}_{final_image_id}_{timestamp}.jpg")
        cv2.imwrite(save_path, frame)

        return final_image_id, frame
    else:
        return "NA", None

def stitch_images(image_dir, save_stitched_folder, save_stitched_path):
    """
    Stitch the latest images into two rows of up to 4 images each.

    Inputs
    ------
    image_dir: str - the directory where images are stored
    save_stitched_folder: str - the folder to save the stitched image
    save_stitched_path: str - the path to save the stitched image

    Returns
    -------
    str - path to the saved stitched image
    """

    images = []
    latest_images = {}
    min_obstacle_id = 1
    max_obstacle_id = 8

    if not os.path.exists(save_stitched_folder):
        os.makedirs(save_stitched_folder)

    # Find the latest images for each obstacle within the specified range
    for filename in os.listdir(image_dir):
        parts = filename.split('_')
        if len(parts) < 3:
            continue  # Skip files that don't match the expected format

        try:
            obstacle_id = int(parts[0])
            timestamp_str = parts[2].replace('.jpg', '')
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d-%H-%M-%S')
        except (ValueError, IndexError):
            continue

        if min_obstacle_id <= obstacle_id <= max_obstacle_id:
            if obstacle_id not in latest_images or latest_images[obstacle_id]['timestamp'] < timestamp:
                latest_images[obstacle_id] = {'timestamp': timestamp, 'filename': filename}

    # Load the latest images
    for obstacle_id in sorted(latest_images.keys()):
        img_path = os.path.join(image_dir, latest_images[obstacle_id]['filename'])
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)

    # Ensure there are at least 2 images to stitch
    if len(images) < 2:
        return "Error: At least two images are required for stitching."

    # Split images into two rows with up to 4 images per row
    row1_images = images[:4]  # First 4 images for the first row
    row2_images = images[4:8]  # Next 4 images for the second row

    # Horizontally concatenate the images in each row
    if len(row1_images) > 1:
        row1_stitched = cv2.hconcat(row1_images)
    else:
        row1_stitched = row1_images[0]  # If only one image in the row

    if row2_images:
        if len(row2_images) > 1:
            row2_stitched = cv2.hconcat(row2_images)
        else:
            row2_stitched = row2_images[0]  # If only one image in the row
    else:
        row2_stitched = None

    # Vertically concatenate the two rows if both rows exist
    if row2_stitched is not None:
        stitched_image = cv2.vconcat([row1_stitched, row2_stitched])
    else:
        stitched_image = row1_stitched

    # Save the stitched image
    save_path = os.path.join(save_stitched_folder, save_stitched_path)
    cv2.imwrite(save_path, stitched_image)

    return save_path

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

