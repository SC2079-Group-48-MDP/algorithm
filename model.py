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

task1_mapping = {
    0: '11', 1: '12', 2: '13', 3: '14', 4: '15', 5: '16', 6: '17', 7: '18', 8: '19', 9: '20',
    10: '21', 11: '22', 12: '23', 13: '24', 14: '25', 15: '26', 16: '27', 17: '28', 18: '29',
    19: '30', 20: '31', 21: '32', 22: '33', 23: '34', 24: '35', 25: '36', 26: '37', 27: '38',
    28: '39', 29: '40', 30: '41'
}

task2_mapping = {
    'BullsEye': 11, 'Left': 20,'Right': 21
}

name_to_id = {
    -1: 'NA', 41: "BullsEye", 11: "1", 12: "2", 13: "3", 14: "4", 15: "5",
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


def display_image(frame, window_name, width=None, height=None):
    """Display the image in a window that does not close automatically and allows resizing."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # If width and height are specified, resize the window
    if width is not None and height is not None:
        cv2.resizeWindow(window_name, width, height)
    else: 
        cv2.resizeWindow(window_name, 1200, 1000)
    
    cv2.imshow(window_name, frame)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(10000)
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
        if class_id == 30:
            return "NA", None

        image_id = task1_mapping.get(class_id, -1)
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
    
def predict_image2(image_bytes, obstacle_id, model):
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
            return None

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
            return None

        # Process the selected result
        x1, y1, x2, y2 = map(int, selected_box.xyxy[0])
        class_id = int(selected_box.cls.item())
        # Reverse lookup the class name from yolo_image_mapping
        class_name = next((key for key, value in task2_mapping.items() if value == class_id), "Unknown")
        image_id = name_to_id.get(class_name, "NA")

        label_text = f"{class_name}, Image ID: {image_id}"

        # Draw label
        frame = draw_label(frame, x1, y1, x2, y2, label_text)

        # Save the annotated frame
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_path = os.path.join(SAVE_DIR, f"{obstacle_id}_{image_id}_{timestamp}.jpg")
        cv2.imwrite(save_path, frame)

        return class_name
    else:
        return None

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

    for filename in os.listdir(image_dir):
        parts = filename.split('_')
        if len(parts) < 3:
            continue
        try:
            obstacle_id = int(parts[0])
            timestamp_str = parts[2].replace('.jpg', '')
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d-%H-%M-%S')
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
        else:
            print(f"Failed to load image: {img_path}")

    if len(images) < 2:
        return None  # Return None to signal failure in stitching

    row_count = 3 if len(images) in [5,6] else 4
    first_image = images[0]
    img_height, img_width = first_image.shape[:2]
    blank_image = create_blank_image(img_width, img_height)

    if len(images) < row_count*2:
        for _ in range(row_count*2 - len(images)):
            images.append(blank_image)

    row1_images = images[:row_count]
    row2_images = images[row_count: row_count*2] if len(images) > 4 else None

    row1_stitched = cv2.hconcat(row1_images) if len(row1_images) > 1 else row1_images[0]
    row2_stitched = cv2.hconcat(row2_images) if row2_images and len(row2_images) > 1 else row2_images[0] if row2_images else None

    stitched_image = cv2.vconcat([row1_stitched, row2_stitched]) if row2_stitched is not None else row1_stitched

    if stitched_image is None:
        print("Stitching failed.")
        return None

    save_path = os.path.join(save_stitched_folder, save_stitched_path)
    cv2.imwrite(save_path, stitched_image)

    return stitched_image

def stitch_images2(image_dir, save_stitched_folder, save_stitched_path):
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
    latest_images = {"small": None, "big": None}

    if not os.path.exists(save_stitched_folder):
        os.makedirs(save_stitched_folder)

    for filename in os.listdir(image_dir):
        parts = filename.split('_')
        if len(parts) < 3:
            continue
        try:
            obstacle_id = parts[0]
            timestamp_str = parts[2].replace('.jpg', '')
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d-%H-%M-%S')
        except (ValueError, IndexError):
            continue

        if obstacle_id in latest_images:
            if latest_images[obstacle_id] is None or latest_images[obstacle_id]['timestamp'] < timestamp:
                latest_images[obstacle_id] = {"timestamp": timestamp, 'filename': filename}

    for obstacle_id, data in latest_images.items():
        if data is not None: 
            img_path = os.path.join(image_dir, data["filename"])
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else: print(f"Failed to load image: {img_path}")

    if len(images) < 2:
        print("Not enough images to stitch.")
        return None  # Return None to signal failure in stitching

    stitched_image = cv2.hconcat(images)
    
    if stitched_image is None:
        print("Stitching failed.")
        return None

    save_path = os.path.join(save_stitched_folder, save_stitched_path)
    cv2.imwrite(save_path, stitched_image)

    return stitched_image


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

