import cv2
import mediapipe as mp
import numpy as np
import json

HEIGHT = 720
WIDTH = 720

def map_to_range(number):
    # Ensure the input number is within the range -1 to 1
    if number < -1:
        number = -1
    elif number > 1:
        number = 1
    
    # Map the number to the range 0 to 1
    mapped_number = (number + 1) / 2
    return mapped_number

def jsonToVid(data):
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture('break_31.avi')

    output_file = 'outputFromMobileJson.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 30, (WIDTH, HEIGHT))

    ret, frame = cap.read()
    results = pose.process(frame)

    l = []

    for dt1 in data:
        black_screen = np.zeros((WIDTH,HEIGHT, 3), dtype=np.uint8)
        i = 0
        for dt in dt1:
            results.pose_landmarks.landmark[i].x = map_to_range(dt['x'])
            results.pose_landmarks.landmark[i].y = map_to_range(dt['y'])
            results.pose_landmarks.landmark[i].z = map_to_range(dt['z'])
            i = i + 1

        mp_drawing.draw_landmarks(black_screen, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec((255, 0, 0), 2, 3),
                            mp_drawing.DrawingSpec((255, 0, 255), 2, 3)
                            )
        
        black_screen = np.rot90(black_screen, k=1)
        cv2.imshow('MediaPipe Pose', black_screen)

        cv2.waitKey(0)

        # break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out.write(black_screen)

    cap.release()
    cv2.destroyAllWindows()
    return output_file

with open("formattedMobileLandmarks.json") as file:
    data1 = json.load(file)
    
print(jsonToVid(data1))