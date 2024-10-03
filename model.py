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
    0: '11', 1: '12', 2: '13', 3: '14', 4: '15', 5: '16', 6: '17', 7: '18', 8: '19', 9: '20',
    10: '21', 11: '22', 12: '23', 13: '24', 14: '25', 15: '26', 16: '27', 17: '28', 18: '29',
    19: '30', 20: '31', 21: '32', 22: '33', 23: '34', 24: '35', 25: '36', 26: '37', 27: '38',
    28: '39', 29: '40', 30: '41'
}

name_to_id = {
    'NA': 'NA', 41: "BullsEye", 11: "1", 12: "2", 13: "3", 14: "4", 15: "5",
    16: "6", 17: "7", 18: "8", 19: "9", 20: "A", 21: "B", 22: "C", 23: "D", 24: "E",
    25: "F", 26: "G", 27: "H", 28: "S", 29: "T", 30: "U", 31: "V", 32: "W", 33: "X",
    34: "Y", 35: "Z", 36: "Up", 37: "Down", 38: "Right", 39: "Left", 40: "Stop"
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

    label_position = (10,130)

    # Draw bounding box and label
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 20)
    cv2.putText(image, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
    return image


def predict_image(image_bytes, obstacle_id, model):
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
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            save_path = os.path.join(SAVE_DIR, f"{obstacle_id}_{timestamp}.jpg")
            cv2.imwrite(save_path, frame)
            return "NA", None

        # Find the largest bounding box
        max_area = 0
        max_height = 0
        selected_box = None
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Finding the largest bounding box
            # area = (x2 - x1) * (y2 - y1)
            # if area > max_area:
            #     max_area = area
            #     selected_box = box

            # Finding the tallest bounding box
            height = abs(y1 - y2)
            if height > max_height:
                max_height = height
                selected_box = box


        if selected_box is None:
            return "NA", None

        # Process the selected result
        x1, y1, x2, y2 = map(int, selected_box.xyxy[0])
        class_id = int(selected_box.cls.item())

        image_id = yolo_image_mapping.get(class_id, "NA")
        class_name = name_to_id.get(int(image_id), "NA")
        label_text = f"{class_name}, Image ID: {image_id}"

        # Draw label
        frame = draw_label(frame, x1, y1, x2, y2, label_text)

        # Save the annotated frame
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_path = os.path.join(SAVE_DIR, f"{obstacle_id}_{image_id}_{timestamp}.jpg")
        cv2.imwrite(save_path, frame)

        return image_id, frame
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

    num_images = len(images)
    row_count = 3 if num_images in [5,6] else 4

    first_image = images[0]
    img_height, img_width = first_image.shape[:2]
    blank_image = create_blank_image(img_width, img_height)

    if num_images < row_count*2:
        for _ in range(row_count*2-num_images):
            images.append(blank_image)

    row1_images = images[:row_count]
    row2_images = images[row_count: row_count*2] if num_images > 4 else None

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

def create_blank_image(width, height, channels=3, color=(0,0,0)):
    blank_image = np.zeros((height, width,channels), dtype=np.uint8)
    blank_image[:] = color
    return blank_image

def stitch_image_own():
    """
    Stitches the images in the folder together and saves it into own_results folder

    Basically similar to stitch_image() but with different folder names and slightly different drawing of bounding boxes and text
    """
    imgFolder = 'own_results'
    stitchedPath = os.path.join(imgFolder, f'stitched-{int(time.time())}.jpeg')

    imgPaths = glob.glob(os.path.join(imgFolder + "/annotated_image_*.jpg"))
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

