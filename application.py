import cv2
import mediapipe as mp
import numpy as np
import time
import json
import requests

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

cap= cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = 480
height = 480
rescale_width = 720
rescale_height = 720

writer= cv2.VideoWriter('basicvideo.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (rescale_width,rescale_height))

left = (frame_width - width) // 2
top = (frame_height - height) // 2
right = left + width
bottom = top + height

start_time = time.time()

while True:
    elapsed_seconds = int(time.time() - start_time)

    ret,frame= cap.read()

    frameToRec = frame

    frame = cv2.flip(frame, 1)
    frameToRec = cv2.flip(frameToRec, 1)

    # frame = cv2.resize(frame, (720, 720))
    # frameToRec = cv2.resize(frameToRec, (720, 720))

    if(elapsed_seconds >= 10 ):
        break
    
    if elapsed_seconds >5 and elapsed_seconds < 10:
        # Add overlay text with the elapsed seconds
        text = "Elapsed seconds: {} : recording.....".format(elapsed_seconds)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cropped_frame = frameToRec[top:bottom, left:right]
        resized_frame = cv2.resize(cropped_frame, (rescale_width, rescale_height))
        writer.write(resized_frame)

    if elapsed_seconds <= 5:
        # Add overlay text with the elapsed seconds
        text = "Elapsed seconds: {} : get ready.....".format(elapsed_seconds)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('frame', frame)

    # Check if the user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

# identifying the skeleton

cap= cv2.VideoCapture("basicvideo.avi")

landmark_list = []

dumb = None

while cap.isOpened():
    ret,frameToIdentify= cap.read()
    # print(ret)

    if not ret:
        break

    results = pose.process(frameToIdentify)

    mp_drawing.draw_landmarks(frameToIdentify, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
    pose_landmarks = []
    if results.pose_landmarks is not None:
        for landmark in results.pose_landmarks.landmark:
            pose_landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            })
    else:
        pose_landmarks.append({
            'x': 0,
            'y': 0,
            'z': 0
        })

    # Append the pose landmarks to the list
    landmark_list.append(pose_landmarks)

    cv2.imshow('frame', frameToIdentify)

    cv2.waitKey(1)

payload = json.dumps(landmark_list)

cap.release()
cv2.destroyAllWindows()

headers = {'Content-Type': 'application/json'}

# Send the POST request
response = requests.post("http://4.247.22.145:8000/endpoint", data=payload, headers=headers)
print("sent")

# Check the response status code
if response.status_code == 200:
    print("Request successful!")
    print(response.content)
else:
    print("Request failed with status code:", response.status_code)