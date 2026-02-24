# MediaPipe wrappers
import cv2 as cv
import mediapipe as mp
from mp.tasks import python
from mp.tasks.python import vision


cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv.flip(frame, 1, frame)  # Flip horizontally for a mirror effect

    # BGRA to RGB
    # cv.cvtColor(frame, cv.COLOR_BGR2RGB, frame)


    cv.imshow('Webcam Feed', frame)
    
    if cv.waitKey(1) & 0xFF == 27:  # Press 'ESC' to quit
        break

cap.release()
cv.destroyAllWindows()