from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import JSONResponse
from algo.algo import MazeSolver
from helper import command_generator
import os
import time
from fastapi.middleware.cors import CORSMiddleware
from model import *
from ultralytics import YOLO
from threading import Thread
from datetime import datetime


app = FastAPI()
model = YOLO("./best_task2_pruned.pt")
model.to('cuda')

# Add CORS middleware for communicating server requests through different protocols
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from all origins. You can specify certain domains here.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE)
    allow_headers=["*"],  # Allows all headers
)

@app.get("/status")
def status():
    """
    This is a health check endpoint to check if the server is running
    :return: a json object with a key "result" and value "ok"
    """
    return JSONResponse ({"result": "ok"})

# When endpoint is called, takes in dictionary of values/parameters to be used in calculatiing the path finding algorithm
# Outputs a JSON representing the results of pathfinding calculations
@app.post("/path")
def path_finding(content: dict):

    """
    This is the main endpoint for the path finding algorithm
    :return: a json object with a key "data" and value a dictionary with keys "distance", "path", and "commands"
    """

    # Get the obstacles, big_turn, retrying, robot_x, robot_y, and robot_direction from the json data
    obstacles = content['obstacles']
    # big_turn = int(content['big_turn'])
    retrying = content['retrying']
    robot_x, robot_y = content['robot_x'], content['robot_y']
    robot_direction = int(content['robot_dir'])

    # Initialize MazeSolver object with robot size of 20x20, bottom left corner of robot at (1,1), facing north, and whether to use a big turn or not.
    maze_solver = MazeSolver(20, 20, robot_x, robot_y, robot_direction, big_turn=None)

    # Add each obstacle into the MazeSolver. Each obstacle is defined by its x,y positions, its direction, and its id
    for ob in obstacles:
        maze_solver.add_obstacle(ob['x'], ob['y'], ob['d'], ob['obstacleNumber'])

    start = time.time()
    # Get shortest path
    optimal_path, distance = maze_solver.get_optimal_order_dp(retrying=retrying)
    print(f"Time taken to find shortest path using A* search: {time.time() - start}s")
    print(f"Distance to travel: {distance} units")
    
    # Based on the shortest path, generate commands for the robot
    commands = command_generator(optimal_path, obstacles)

    # Get the starting location and add it to path_results
    path_results = [optimal_path[0].get_dict()]

    # Process each command individually and append the location the robot should be after executing that command to path_results
    i = 0
    for command in commands:
        if command.startswith("SNAP"):
            continue
        if command.startswith("FN"):
            continue
        elif command.startswith("FW") or command.startswith("FS"):
            i += int(command[2:]) // 10
        elif command.startswith("BW") or command.startswith("BS"):
            i += int(command[2:]) // 10
        else:
            i += 1
        path_results.append(optimal_path[i].get_dict())

    if len(commands) == 1 and commands[0] == "FN":
        commands = ["BW10", f"SNAP{obstacles[0]["obstacleNumber"]}", "FN"]

    print(commands)

    # Sends parameters from pathfinding to /path on API server as a JSON
    return JSONResponse({
        "data": {
            'distance': distance,
            'path': path_results,
            'commands': commands
        },
        "error": None
    })

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
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

@app.post("/image")
async def image_predict(files: UploadFile = File(...), obstacle_id: str = Form(...), signal: str = Form(...)):
    #filename = files.filename

    image_bytes = await files.read()
    # Add more debugging or validation here
    #print(f"Received obstacle_id: {obstacle_id}, file size: {len(image_bytes)} bytes")
    #image_id = predict_image(filename, model, signal)

    class_name = predict_image2(image_bytes, obstacle_id, model)

    if class_name is not None:
        # Sends identifiers to /image on API server as a JSON
        result = {
            "obstacle_id": obstacle_id,
            "class_name": class_name,
            # "stop": image_id != 10 #For checklist (Navigating around the obstacle)
        }
        
    else:
        print("Prediction failed or image could not be processed.")
        result = {
            "obstacle_id": obstacle_id,
            "class_name": "NA"
        }
    
    return JSONResponse(content=result)


# For stiching together images when called
@app.get("/stitch")
def stitch():
    image_dir = SAVE_DIR
    save_stitched_folder = "./stitched_image"
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_stitched_path = f"{timestamp}_stitched_image.jpg"
    # return stitched image
    stitched_image = stitch_images2(image_dir, save_stitched_folder, save_stitched_path)
    if stitched_image is not None:
        #print(f"Stitched image shape: {stitched_image.shape}")
        display_image(stitched_image, f"Stitched Image")
    else:
        return JSONResponse({"result": "failed"})

    # save_stitched_own_path = "stitched_image_own.jpg"
    # img2 = stitch_image_own(image_dir, save_stitched_own_path)
    # if img2:
    #     display_image(img2, "Stitched Image (Own)")

    # Return a response to show that the image stitching process 
    return JSONResponse({"result": "success"})

