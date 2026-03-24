"""IrisFlow — real-time iris gaze tracker with gesture control.

Entry point.  Run with:
    python -m src          # as package
    irisflow               # if installed via pip
"""

import time
import platform

import cv2 as cv
import pyautogui
import numpy as np

# Safety: moving mouse to corner (0,0) aborts the program
pyautogui.FAILSAFE = True
# Disable pyautogui's built-in 0.1 s pause after every call
pyautogui.PAUSE = 0

from src.core.tracker import VisionTracker
from src.core.processor import GazeProcessor
from src.core.calibration import GazeCalibrator
from src.gestures.gestures import GestureController

# ── Constants ────────────────────────────────────────────────────────
CAM_WIDTH, CAM_HEIGHT = 640, 480
HEAD_COMP_SCALE = 0.012       # Multiplier for head-pose compensation
SCROLL_SENSITIVITY = 15       # pyautogui scroll clicks per gesture frame
DRIFT_INTERVAL_S = 30         # Auto drift-correct every N seconds


# ── Calibration phase ────────────────────────────────────────────────

def run_calibration(cap, tracker, screen_w, screen_h):
    """Interactive 9-point calibration.

    Draws dots on a full-screen window; user looks at each and presses
    SPACE to capture.  Returns a GazeCalibrator with a computed homography,
    or None if the user pressed ESC.
    """
    calibrator = GazeCalibrator()
    point_idx = 0

    print("\n--- Starting Calibration ---")
    print("Look at the RED DOT and press SPACE for each point (9 total).")

    cv.namedWindow("Calibration", cv.WINDOW_NORMAL)
    cv.moveWindow("Calibration", 0, 0)
    cv.resizeWindow("Calibration", screen_w, screen_h)

    while point_idx < 9:
        ret, frame = cap.read()
        if not ret:
            break

        bg = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        target_x, target_y = calibrator.get_current_point(screen_w, screen_h, point_idx)

        # Target dot: red ring + white centre
        cv.circle(bg, (target_x, target_y), 15, (0, 0, 255), -1)
        cv.circle(bg, (target_x, target_y), 5, (255, 255, 255), -1)
        cv.putText(
            bg,
            f"Point {point_idx + 1}/9: Look here and press SPACE",
            (screen_w // 2 - 250, screen_h // 2 + 100),
            cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        cv.imshow("Calibration", bg)

        face_res, _ = tracker.process(frame)
        gaze_ratio = tracker.get_gaze_ratio(face_res)

        key = cv.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            if gaze_ratio:
                calibrator.add_calibration_point(
                    target_x, target_y, gaze_ratio[0], gaze_ratio[1],
                )
                print(
                    f"  Captured {point_idx + 1}: "
                    f"Screen({target_x}, {target_y}) -> "
                    f"Gaze({gaze_ratio[0]:.3f}, {gaze_ratio[1]:.3f})"
                )
                point_idx += 1
            else:
                print("  Eye not detected — please face the camera.")
        elif key == 27:  # ESC
            cv.destroyWindow("Calibration")
            return None

    cv.destroyWindow("Calibration")

    print("Computing homography (RANSAC) ...")
    if calibrator.calculate_mapping():
        print("Calibration successful.")
    else:
        print("Calibration failed (not enough good points). Using fallback.")

    return calibrator


# ── Main loop ────────────────────────────────────────────────────────

def main():
    screen_w, screen_h = pyautogui.size()

    # 1. Initialise modules
    try:
        tracker = VisionTracker()
    except FileNotFoundError as exc:
        print(f"Model error: {exc}")
        return

    gestures = GestureController(wink_threshold=0.06, cooldown_ms=500)
    processor = GazeProcessor(min_cutoff=0.8, beta=0.02)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        return

    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    # 2. Calibration
    calibrator = run_calibration(cap, tracker, screen_w, screen_h)
    if calibrator is None:
        cap.release()
        return

    print(f"IrisFlow active — screen {screen_w}x{screen_h}")

    # FPS tracking
    fps_t0 = time.time()
    frame_count = 0
    fps_display = 0.0

    # Drift correction timer
    last_drift_t = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv.flip(frame, 1, frame)

        # ── FPS measurement ──────────────────────────────────────
        frame_count += 1
        elapsed = time.time() - fps_t0
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            fps_t0 = time.time()

        # ── Tracking ─────────────────────────────────────────────
        face_res, hand_res = tracker.process(frame)

        gaze_ratio = tracker.get_gaze_ratio(face_res)
        head_pose = tracker.get_head_pose(face_res)

        if gaze_ratio and head_pose:
            # Head compensation
            comp_x = gaze_ratio[0] - head_pose[0] * HEAD_COMP_SCALE
            comp_y = gaze_ratio[1] - head_pose[1] * HEAD_COMP_SCALE

            # Calibration transform (homography)
            raw_sx, raw_sy = calibrator.apply_transform(comp_x, comp_y)

            # One-Euro smoothing
            sx, sy = processor.process(raw_sx, raw_sy)

            if sx is not None and sy is not None:
                sx = max(0, min(sx, screen_w - 1))
                sy = max(0, min(sy, screen_h - 1))

                # ── Move cursor ──────────────────────────────────
                pyautogui.moveTo(sx, sy)

                # ── Wink → click ─────────────────────────────────
                l_blink, r_blink = tracker.get_blink_scores(face_res)
                wink = gestures.detect_wink(l_blink, r_blink)

                if wink == "left":
                    pyautogui.click()
                    cv.putText(frame, "LEFT CLICK", (240, 400),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif wink == "right":
                    pyautogui.rightClick()
                    cv.putText(frame, "RIGHT CLICK", (240, 400),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # ── Hand gestures ────────────────────────────────
                hand_lm = tracker.get_hand_landmarks(hand_res)
                if hand_lm is not None:
                    finger_count = tracker.count_extended_fingers(hand_lm)
                    gesture_name = gestures.classify_hand_gesture(hand_lm, finger_count)

                    if gesture_name == "scroll":
                        delta = gestures.detect_scroll(hand_lm)
                        if delta is not None:
                            pyautogui.scroll(int(-delta * SCROLL_SENSITIVITY))
                            cv.putText(frame, "SCROLL", (10, 460),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    elif gesture_name == "zoom":
                        pinch_delta = gestures.detect_pinch(hand_lm)
                        if pinch_delta is not None:
                            # Zoom via Cmd+scroll (macOS) or Ctrl+scroll (other)
                            mod = "command" if platform.system() == "Darwin" else "ctrl"
                            direction = 1 if pinch_delta > 0 else -1
                            with pyautogui.hold(mod):
                                pyautogui.scroll(direction)
                            cv.putText(frame, "ZOOM", (10, 460),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    elif gesture_name == "drag":
                        # 3 fingers: hold mouse button (drag initiated by iris)
                        pyautogui.mouseDown()
                        cv.putText(frame, "DRAG", (10, 460),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                    elif gesture_name == "switch_desktop":
                        swipe = gestures.detect_swipe(hand_lm)
                        if swipe == "left":
                            pyautogui.hotkey("ctrl", "left")
                            cv.putText(frame, "DESKTOP <-", (10, 460),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                        elif swipe == "right":
                            pyautogui.hotkey("ctrl", "right")
                            cv.putText(frame, "DESKTOP ->", (10, 460),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                    elif gesture_name == "mission_control":
                        if platform.system() == "Darwin":
                            pyautogui.hotkey("ctrl", "up")
                        else:
                            pyautogui.hotkey("super", "tab")
                        cv.putText(frame, "MISSION CTRL", (10, 460),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                else:
                    # No hand detected → ensure mouse button released
                    pyautogui.mouseUp()

                # ── Drift correction (periodic) ──────────────────
                now = time.time()
                if now - last_drift_t > DRIFT_INTERVAL_S:
                    screen_cx, screen_cy = screen_w / 2, screen_h / 2
                    calibrator.correct_drift(
                        (screen_cx, screen_cy), (comp_x, comp_y),
                    )
                    last_drift_t = now

                # ── Overlay ──────────────────────────────────────
                cv.circle(
                    frame,
                    (int(gaze_ratio[0] * CAM_WIDTH), int(gaze_ratio[1] * CAM_HEIGHT)),
                    4, (0, 255, 0), -1,
                )
                cv.putText(
                    frame, f"Mouse: {int(sx)}, {int(sy)}",
                    (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,
                )

        # FPS overlay (always visible)
        cv.putText(
            frame, f"FPS: {fps_display:.1f}",
            (CAM_WIDTH - 130, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

        cv.imshow("IrisFlow Feed", frame)
        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
