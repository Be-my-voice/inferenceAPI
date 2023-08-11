import cv2
import mediapipe as mp
import numpy as np
import json

def jsonToVid(data):
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture('sit_31.mp4')

    output_file = 'outputFromMobileJson.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 30, (720, 720))

    ret, frame = cap.read()
    results = pose.process(frame)

    l = []

    for dt1 in data:
        black_screen = np.zeros((720,720, 3), dtype=np.uint8)
        i = 0
        for dt in dt1:
            results.pose_landmarks.landmark[i].x = dt['x']
            results.pose_landmarks.landmark[i].y = dt['y']
            results.pose_landmarks.landmark[i].z = dt['z']
            i = i + 1

        mp_drawing.draw_landmarks(black_screen, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec((255, 0, 0), 2, 3),
                            mp_drawing.DrawingSpec((255, 0, 255), 2, 3)
                            )
        
        cv2.imshow('MediaPipe Pose', black_screen)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out.write(black_screen)

    cap.release()
    cv2.destroyAllWindows()
    return output_file

with open("formattedMobileLandmarks.json") as file:
    data1 = json.load(file)
    
print(jsonToVid(data1))