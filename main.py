from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from algo.algo import MazeSolver
from helper import command_generator
import os
import time
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from model import *


app = FastAPI()
model = None

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
        maze_solver.add_obstacle(ob['x'], ob['y'], ob['d'], ob['id'])

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
        if command.startswith("FIN"):
            continue
        elif command.startswith("FW") or command.startswith("FS"):
            i += int(command[2:]) // 10
        elif command.startswith("BW") or command.startswith("BS"):
            i += int(command[2:]) // 10
        else:
            i += 1
        path_results.append(optimal_path[i].get_dict())

    # Sends parameters from pathfinding to /path on API server as a JSON
    return JSONResponse({
        "data": {
            'distance': distance,
            'path': path_results,
            'commands': commands
        },
        "error": None
    })

# When called, will process the image and identify which of the known images it is
# Outputs known image id as JSON
@app.post("/image")
async def image_predict(file: UploadFile = File(...)):
    filename = file.filename
    file_location = f"uploads/{filename}"
    with open(file_location, "wb") as f:
        f.write(file.file.read())
    constituents = filename.split("_")
    obstacle_id = constituents[1]

    ## Week 8 ## 
    #signal = constituents[2].strip(".jpg")
    #image_id = predict_image(filename, model, signal)

    image_id = predict_image_week_9(file_location, model)

    # Sends identifiers to /image on API server as a JSON
    result = {
        "obstacle_id": obstacle_id,
        "image_id": image_id
    }

    # Returns image id as well as obstacle in JSON format
    return JSONResponse(content=result)


# For stiching together images when called
@app.get("/stitch")
def stitch():
    img = stitch_image()
    img.show()
    img2 = stitch_image_own()
    img2.show()

    # Return a response to show that the image stitching process 
    return JSONResponse({"result": "ok"})

#if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

#if __name__ == '__main__':
    #import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=5000)

