import cv2
import time

# Define the video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Get the actual frame rate from the camera
actual_fps = cap.get(cv2.CAP_PROP_FPS)

# Define the desired duration and calculate the number of frames needed
duration = 3  # Duration in seconds
desired_frames = int(30 * duration)

# Create an output video file
output_video = cv2.VideoWriter('output.avi', fourcc, actual_fps, (640, 480))

# Start capturing frames
for _ in range(desired_frames):
    # Read a frame from the camera
    ret, frame = cap.read()

    # Write the frame to the output video
    output_video.write(frame)

    # Display the frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

    time.sleep(0.333)

# Release the resources
cap.release()
output_video.release()
cv2.destroyAllWindows()
