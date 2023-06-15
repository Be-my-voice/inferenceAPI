import cv2
import time

cap= cv2.VideoCapture(0)

writer= cv2.VideoWriter('basicvideo.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (720,720))

start_time = time.time()

while True:
    ret,frame= cap.read()

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    frame = cv2.resize(frame, (720, 720))

    height, width, _ = frame.shape

    writer.write(frame)

    # calculating elapsed time
    elapsed_seconds = int(time.time() - start_time)

    # Add overlay text with the elapsed seconds
    text = "Elapsed seconds: {}".format(elapsed_seconds)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('frame', frame)

    # Check if the user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
