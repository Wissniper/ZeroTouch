import cv2 as cv
import sys
import os
import pyautogui

# Safety Mandate
pyautogui.FAILSAFE = True

# Add parent dir to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.tracker import VisionTracker
from core.mapper import CoordinateMapper
from core.processor import GazeProcessor
from gestures.gestures import GestureController

def main():
    # Performance Constants
    CAM_WIDTH, CAM_HEIGHT = 640, 480

    # Head compensation strength.
    # get_head_pose() returns matrix[0,2] / matrix[1,2] (≈ sin of yaw/pitch, range -1..1).
    # A 30° head turn → value ≈ 0.5. Scale maps that to gaze-ratio units (0..1).
    # Increase if cursor still drifts on head movement; decrease if over-corrected.
    HEAD_COMP_SCALE = 0.4

    # 1. Init Modules
    screen_w, screen_h = pyautogui.size()
    mapper = CoordinateMapper(screen_w, screen_h, margin=0.35)
    tracker = VisionTracker()
    gestures = GestureController(wink_threshold=0.06)

    # One-Euro filter: min_cutoff controls base smoothing (lower = smoother, more lag).
    # beta controls responsiveness during fast eye movements (higher = less lag).
    processor = GazeProcessor(min_cutoff=1.0, beta=0.05)

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    print(f"IrisFlow Gaze Mode (Smoothed). Tracking on {screen_w}x{screen_h}.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        cv.flip(frame, 1, frame)
        
        # 2. Tracking
        face_res, hand_res = tracker.process(frame)
        
        # 3. Gaze Processing & Head Compensation
        gaze_ratio = tracker.get_gaze_ratio(face_res)
        head_pose = tracker.get_head_pose(face_res)
        
        if gaze_ratio and head_pose:
            # Head Compensation: subtract the rotation contribution from the raw gaze ratio.
            # head_pose = (yaw, pitch) where yaw = matrix[0,2] ≈ sin(horizontal turn).
            # Positive yaw means face turns RIGHT in the flipped image (= user turns LEFT),
            # which biases gaze_ratio[0] upward → cursor drifts right. Subtracting corrects this.
            comp_x = gaze_ratio[0] + head_pose[0] * HEAD_COMP_SCALE
            comp_y = gaze_ratio[1] - head_pose[1] * HEAD_COMP_SCALE

            # 4. Smoothing (Exponential Moving Average)
            smooth_x, smooth_y = processor.process(comp_x, comp_y)

            # Map the smoothed and compensated gaze to the screen
            sx, sy = mapper.map_to_screen(smooth_x, smooth_y)
            
            pyautogui.moveTo(sx, sy, _pause=False)
            
            # 5. Wink Detection
            l_blink, r_blink = tracker.get_blink_scores(face_res)
            wink = gestures.detect_wink(l_blink, r_blink)
            
            if wink == 'left':
                cv.putText(frame, "LEFT CLICK!", (240, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                # pyautogui.click()
            elif wink == 'right':
                cv.putText(frame, "RIGHT CLICK!", (240, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                # pyautogui.rightClick()

            # Debug Overlay
            cv.putText(frame, f"Raw Gaze: X:{gaze_ratio[0]:.2f} Y:{gaze_ratio[1]:.2f}", (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
            cv.putText(frame, f"Smoothed: X:{smooth_x:.2f} Y:{smooth_y:.2f}", (10, 60), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
            cv.putText(frame, f"Head Yaw:{head_pose[0]:.2f} Pitch:{head_pose[1]:.2f}", (10, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 1)

        # 6. UI Feed
        cv.imshow('IrisFlow', frame)
        if cv.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
