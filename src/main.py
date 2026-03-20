import cv2 as cv
import sys
import os
import pyautogui
import numpy as np

# Safety Mandate
pyautogui.FAILSAFE = True

# Add parent dir to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.tracker import VisionTracker
from core.processor import GazeProcessor
from core.calibration import GazeCalibrator
from gestures.gestures import GestureController

def run_calibration(cap, tracker, screen_w, screen_h):
    """
    Interactive 9-point calibration phase.
    Draws dots on the screen; user looks and presses SPACE for each.
    """
    calibrator = GazeCalibrator()
    point_idx = 0
    
    print("\n--- Starting Calibration ---")
    print("Look at the RED DOT and press SPACE for each point (9 total).")

    # Create a screen-sized window for calibration (avoids macOS fullscreen space transition)
    cv.namedWindow("Calibration", cv.WINDOW_NORMAL)
    cv.moveWindow("Calibration", 0, 0)
    cv.resizeWindow("Calibration", screen_w, screen_h)

    while point_idx < 9:
        ret, frame = cap.read()
        if not ret: break

        # Background for the calibration window
        bg = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        
        # Get current target dot coordinates
        target_x, target_y = calibrator.get_current_point(screen_w, screen_h, point_idx)
        
        # Draw the target dot (Red)
        cv.circle(bg, (target_x, target_y), 15, (0, 0, 255), -1)
        cv.circle(bg, (target_x, target_y), 5, (255, 255, 255), -1)
        
        cv.putText(bg, f"Point {point_idx+1}/9: Look here and press SPACE", 
                   (screen_w//2 - 200, screen_h//2 + 100), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv.imshow("Calibration", bg)
        
        # Detect Face and Gaze in background (to capture the gaze_ratio)
        face_res, _ = tracker.process(frame)
        gaze_ratio = tracker.get_gaze_ratio(face_res)

        key = cv.waitKey(1) & 0xFF
        if key == 32: # SPACEBAR
            if gaze_ratio:
                # Store the mapping: Target Screen (x,y) -> Gaze Ratio (x,y)
                calibrator.add_calibration_point(target_x, target_y, gaze_ratio[0], gaze_ratio[1])
                print(f"Captured Point {point_idx+1}: Screen({target_x}, {target_y}) -> Gaze({gaze_ratio[0]:.2f}, {gaze_ratio[1]:.2f})")
                point_idx += 1
            else:
                print("⚠️ Eye not detected! Please look at the camera.")
        elif key == 27: # ESC
            cv.destroyWindow("Calibration")
            return None

    # Calculate the mapping matrix
    print("Calculating Homography Matrix...")
    if calibrator.calculate_mapping():
        print("✅ Calibration Successful!")
    else:
        print("❌ Calibration Failed (Need more points). Using fallback.")

    cv.destroyWindow("Calibration")
    return calibrator

def main():
    # Performance Constants
    CAM_WIDTH, CAM_HEIGHT = 640, 480
    
    # 1. Init Modules
    screen_w, screen_h = pyautogui.size()
    tracker = VisionTracker()
    gestures = GestureController(wink_threshold=0.06)
    processor = GazeProcessor(min_cutoff=0.8, beta=0.02) # Smoother for gaze

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    # 2. RUN CALIBRATION (NEW)
    calibrator = run_calibration(cap, tracker, screen_w, screen_h)
    if calibrator is None: return

    print(f"IrisFlow Active. Smoothing On. Control Screen: {screen_w}x{screen_h}.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        cv.flip(frame, 1, frame)
        
        # 3. Tracking
        face_res, hand_res = tracker.process(frame)
        
        # 4. Gaze Processing with Head Compensation
        gaze_ratio = tracker.get_gaze_ratio(face_res)
        head_pose = tracker.get_head_pose(face_res)
        
        if gaze_ratio and head_pose:
            # Apply Head Compensation
            head_offset_x = head_pose[0] * 0.012 
            head_offset_y = head_pose[1] * 0.012
            comp_x = gaze_ratio[0] - head_offset_x
            comp_y = gaze_ratio[1] - head_offset_y

            # 5. Apply Calibration Transform (REPLACES linear mapper)
            raw_screen_x, raw_screen_y = calibrator.apply_transform(comp_x, comp_y)

            # 6. One-Euro Smoothing
            sx, sy = processor.process(raw_screen_x, raw_screen_y)

            # Clamp to screen bounds
            if sx is not None and sy is not None:
                sx = max(0, min(sx, screen_w - 1))
                sy = max(0, min(sy, screen_h - 1))

                # Move Mouse
                pyautogui.moveTo(sx, sy, _pause=False)

                # 7. Gesture Handling
                l_blink, r_blink = tracker.get_blink_scores(face_res)
                wink = gestures.detect_wink(l_blink, r_blink)
                
                if wink == 'left':
                    cv.putText(frame, "LEFT CLICK", (240, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                elif wink == 'right':
                    cv.putText(frame, "RIGHT CLICK", (240, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                # Overlay
                cv.circle(frame, (int(gaze_ratio[0]*CAM_WIDTH), int(gaze_ratio[1]*CAM_HEIGHT)), 4, (0,255,0), -1)
                cv.putText(frame, f"Mouse: {int(sx)}, {int(sy)}", (10, 30), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

        # 8. UI Feed
        cv.imshow('IrisFlow Feed', frame)
        if cv.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
