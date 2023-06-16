import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

cap= cv2.VideoCapture(0)

writer= cv2.VideoWriter('basicvideo.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (720,720))

start_time = time.time()

while True:
    elapsed_seconds = int(time.time() - start_time)

    ret,frame= cap.read()

    frameToRec = frame

    frame = cv2.flip(frame, 1)
    frameToRec = cv2.flip(frameToRec, 1)

    frame = cv2.resize(frame, (720, 720))
    frameToRec = cv2.resize(frameToRec, (720, 720))

    if(elapsed_seconds >= 9 ):
        break
    
    if elapsed_seconds >5 and elapsed_seconds < 9:
        # Add overlay text with the elapsed seconds
        text = "Elapsed seconds: {} : recording.....".format(elapsed_seconds)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        writer.write(frameToRec)

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

while cap.isOpened():
    ret,frameToIdentify= cap.read()
    # print(ret)

    results = pose.process(frameToIdentify)

    mp_drawing.draw_landmarks(frameToIdentify, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
    cv2.imshow('frame', frameToIdentify)

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()