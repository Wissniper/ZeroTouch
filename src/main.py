import cv2 as cv
import sys
import os

# Ensure the parent directory is in the path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.tracker import VisionTracker, draw_face_landmarks, draw_hand_landmarks

def main():
    # Application entry point for IrisFlow.
    # 1. Initialize Tracker (outside the loop for performance!)
    try:
        tracker = VisionTracker(
            face_model_path='face_landmarker.task',
            hand_model_path='hand_landmarker.task'
        )
    except Exception as e:
        print(f"Error initializing VisionTracker: {e}")
        print("Please ensure face_landmarker.task and hand_landmarker.task are in the root directory.")
        return

    # 2. Setup OpenCV Capture
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv.CAP_PROP_FPS, 30)

    print("IrisFlow started. Press 'ESC' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Mirror the image for intuitive tracking
        cv.flip(frame, 1, frame)

        # 3. Get Landmarks
        face_res, hand_res = tracker.process(frame)

        # 4. Optional Visualization (Draw Landmarks)
        if face_res.face_landmarks:
            frame = draw_face_landmarks(frame, face_res)

        if hand_res.hand_landmarks:
            frame = draw_hand_landmarks(frame, hand_res)

        # 5. TODO: Pass results to core.processor for smoothing/scaling
        
        # Display the feed
        cv.imshow('IrisFlow - Tracking Feed', frame)

        if cv.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    # Clean up
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
