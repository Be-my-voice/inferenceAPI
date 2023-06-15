import cv2
import mediapipe as mp
import numpy as np
import json
from mediapipe.framework.formats import landmark_pb2
from fastapi import FastAPI, Request
from inferModel import *

def jsonToVid(data):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture('sit_31.avi')

    output_file = 'outputFromJson.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, 30, (720, 720))

    ret, frame = cap.read()
    results = pose.process(frame)

    
    
    myFrozenObject = frozenset({(4, 10), (5, 9), (7, 9), (3, 5),  (4, 6), (5, 11), (0, 2),  (4, 8), (5, 7), (1, 3), (6, 8),  (2, 4)})

    for dt1 in data:
        black_screen = np.zeros((720,720, 3), dtype=np.uint8)
        i = 0

        landmark_subset = landmark_pb2.NormalizedLandmarkList(
            landmark = results.pose_landmarks.landmark[11:23])

        for dt in dt1[11:23]:
            landmark_subset.landmark[i].x = dt['x']
            landmark_subset.landmark[i].y = dt['y']
            landmark_subset.landmark[i].z = dt['z']
            i = i + 1

        

        mp_drawing.draw_landmarks(black_screen, landmark_subset, myFrozenObject,
                            mp_drawing.DrawingSpec((255, 0, 0), 15, 2),
                            mp_drawing.DrawingSpec((255, 0, 255), 15, 1)
                            )
        
        cv2.imshow('MediaPipe Pose', black_screen)
        out.write(black_screen)

    cap.release()
    cv2.destroyAllWindows()
    return output_file



app = FastAPI()

@app.on_event("startup")
async def initModel():
    print("Loading model to API.....")
    app.model = InferModel()
    print("Model has been loaded to API")


@app.post("/endpoint")
async def process_payload(request: Request):
    payload = await request.json()
    fileName = jsonToVid(payload)
    prediction = app.model.predict(fileName)
    # Delete the video after prediction
    os.remove(fileName)
    return { "prediction": prediction }