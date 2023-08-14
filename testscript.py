import cv2

# Define the video capture object
video = cv2.VideoCapture(0)

# Get the default video frame width and height
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output.avi', fourcc, 30.0, (frame_width, frame_height))

# Record video for 3 seconds
start_time = cv2.getTickCount()
while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < 3:
    # Read a frame from the video capture
    ret, frame = video.read()

    # Write the frame to the video writer
    output_video.write(frame)

    # Display the frame (optional)
    cv2.imshow('Recording', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and writer objects
video.release()
output_video.release()

# Destroy any OpenCV windows
cv2.destroyAllWindows()
