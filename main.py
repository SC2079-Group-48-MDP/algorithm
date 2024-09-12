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

# Add CORS middleware
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

@app.post("/path")
def path_finding(content: dict):

    """
    This is the main endpoint for the path finding algorithm
    :return: a json object with a key "data" and value a dictionary with keys "distance", "path", and "commands"
    """

    obstacles = content['obstacles']
    retrying = content['retrying']
    robot_x, robot_y = content['robot_x'], content['robot_y']
    robot_direction = int(content['robot_dir'])

    maze_solver = MazeSolver(20, 20, robot_x, robot_y, robot_direction, big_turn=None)

    for ob in obstacles:
        maze_solver.add_obstacle(ob['x'], ob['y'], ob['d'], ob['id'])

    start = time.time()
    optimal_path, distance = maze_solver.get_optimal_order_dp(retrying=retrying)
    print(f"Time taken to find shortest path using A* search: {time.time() - start}s")
    print(f"Distance to travel: {distance} units")
    
    commands = command_generator(optimal_path, obstacles)

    path_results = [optimal_path[0].get_dict()]
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

    return JSONResponse({
        "data": {
            'distance': distance,
            'path': path_results,
            'commands': commands
        },
        "error": None
    })

@app.post("/image")
async def image_predict(file: UploadFile = File(...)):
    filename = file.filename
    file_location = f"uploads/{filename}"
    with open(file_location, "wb") as f:
        f.write(file.file.read())
    constituents = filename.split("_")
    obstacle_id = constituents[1]

    image_id = predict_image_week_9(file_location, model)

    result = {
        "obstacle_id": obstacle_id,
        "image_id": image_id
    }
    return JSONResponse(content=result)

@app.get("/stitch")
def stitch():
    img = stitch_image()
    img.show()
    img2 = stitch_image_own()
    img2.show()
    return JSONResponse({"result": "ok"})

#if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

#if __name__ == '__main__':
    #import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=5000)

