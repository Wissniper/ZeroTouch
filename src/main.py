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
from core.gestures import GestureController

def main():
    # Performance Constants
    CAM_WIDTH, CAM_HEIGHT = 640, 480
    
    # 1. Init Modules
    screen_w, screen_h = pyautogui.size()
    # Higher sensitivity for gaze: we use a smaller margin since gaze ratio is already relative
    mapper = CoordinateMapper(screen_w, screen_h, margin=0.35)
    tracker = VisionTracker()
    gestures = GestureController(wink_threshold=0.06)

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    print(f"IrisFlow Gaze Mode. Tracking on {screen_w}x{screen_h}.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        cv.flip(frame, 1, frame)
        
        # 2. Tracking
        face_res, hand_res = tracker.process(frame)
        
        # 3. Gaze Processing (Relative to Eye Socket)
        gaze_ratio = tracker.get_gaze_ratio(face_res)
        if gaze_ratio:
            # Map the gaze ratio (0-1) to the screen
            sx, sy = mapper.map_to_screen(gaze_ratio[0], gaze_ratio[1])
            
            # Uncomment the next line to actually control the mouse
            pyautogui.moveTo(sx, sy, _pause=False)
            
            # 4. Wink Detection
            l_blink, r_blink = tracker.get_blink_scores(face_res)
            wink = gestures.detect_wink(l_blink, r_blink)
            
            if wink == 'left':
                cv.putText(frame, "LEFT CLICK!", (240, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                # pyautogui.click()
            elif wink == 'right':
                cv.putText(frame, "RIGHT CLICK!", (240, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                # pyautogui.rightClick()

            # Debug Overlay
            # Show the raw gaze ratio and the calculated screen position
            cv.putText(frame, f"Gaze: X:{gaze_ratio[0]:.2f} Y:{gaze_ratio[1]:.2f}", (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
            cv.putText(frame, f"Target Screen: {sx}, {sy}", (10, 60), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
            cv.putText(frame, f"Blinks: L={l_blink:.2f} R={r_blink:.2f}", (10, 90), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 5. UI Feed
        cv.imshow('IrisFlow', frame)
        if cv.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
