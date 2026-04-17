# Repository Map & Pipeline Architecture

## Directory Structure & Key Files

### src/main.py (298 lines)
**Role:** Entry point, main event loop, camera capture, FPS tracking
**Pipeline Stage:** Orchestrator
**Responsibilities:**
- Camera initialization (640×480 @ 30 FPS target)
- Per-frame event loop with sequential processing
- Calibration phase management
- PyAutoGUI action dispatch (cursor movement, clicks, scrolling)
- Overlay rendering for debugging
- FPS counter (rolling 1-second average)

**Key Issues:**
- No profiling/timing instrumentation per stage
- No frame skipping or adaptive processing
- Synchronous capture and detection (blocking I/O)
- Full 5-stage pipeline every frame (see below)
- No confidence gating before action dispatch

### src/core/tracker.py (175 lines)
**Role:** MediaPipe wrapper for face and hand detection
**Responsibilities:**
- Face landmark detection (468 landmarks + blendshapes + transformation matrix)
- Hand landmark detection (21 landmarks per hand)
- Iris coordinate extraction (average of 10 iris landmarks, indices 468-477)
- Gaze ratio computation (normalized iris position within eye socket)
- Blink score extraction (blendshape indices 9, 10)
- Head pose extraction (yaw, pitch from transformation matrix)
- Finger counting heuristic (tip-above-MCP rule)

**Key Issues:**
- Line 167-170: Finger counting uses inconsistent logic (thumb: x-comparison; others: y-comparison)
- No confidence score checks before returning results
- Division-by-zero guards present (_EPS = 1e-6) but only in gaze ratio
- No fallback if iris landmarks are not detected
- No robustness to lighting conditions or occlusions
- Hand detection only returns first hand (num_hands=1); ignores left/right context

### src/core/processor.py (110 lines)
**Role:** One-Euro adaptive smoothing filter
**Responsibilities:**
- Per-axis low-pass filtering with adaptive cutoff
- Initialization on first frame
- Timestamp-based sampling frequency estimation

**Quality:**
- Well-implemented; handles edge cases (dt≤1e-6, bootstrap)
- Good docstring with algorithm explanation
- No issues identified

### src/core/calibration.py (130 lines)
**Role:** 9-point homography calibration with drift correction
**Responsibilities:**
- Collect 9 calibration points (screen target ↔ gaze reading pairs)
- Compute homography via cv2.findHomography (RANSAC, threshold=5.0)
- Apply perspective transform to gaze coordinates
- Translational drift correction (shifts H[0,2] and H[1,2])

**Key Issues:**
- Only translational drift correction (no rotation)
- Homography computation requires ≥4 points; with N=9, outliers can still bias the fit
- No confidence measure on the computed homography
- No validation that inlier ratio is sufficient (just checks inliers < 4)
- No re-calibration without restart

### src/gestures/gestures.py (186 lines)
**Role:** Hand gesture state machine and detectors
**Responsibilities:**
- Wink detection via blendshape differential (left_blink - right_blink)
- Cooldown logic per gesture type
- Pinch detection (thumb-index distance delta)
- Scroll detection (index Y-velocity)
- Swipe detection (index X-velocity)
- Gesture classification (finger count → gesture type)

**Key Issues:**
- `classify_hand_gesture()` (line 169–185): Maps finger count directly without temporal consistency
- No confidence scoring on gesture classification
- Pinch/scroll use raw velocity deltas with fixed micro-noise threshold (0.005)
- Cooldown only applies to repeated *identical* gestures; not to rapidly alternating gestures
- No state machine to prevent gesture transitions within a frame
- No "unknown gesture" or confidence state

### tests/test_*.py (4 test files, 59 tests)
**Quality:** Good unit test coverage for gestures, processor, tracker, calibration
**Gaps:** No integration tests; no performance benchmarks; no robustness tests under varying light/angle

### setup_models.py
**Role:** Downloads MediaPipe face + hand .task models (~11 MB)

---

## Runtime Pipeline (Per Frame)

```
Frame N (time t)
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 1. CAPTURE                                                       │
│    └─ cv.VideoCapture.read() → BGR 640×480 frame               │
│    └─ Flip horizontally (mirror mode)                           │
│    └─ Convert to RGB for MediaPipe                              │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. FACE DETECTION (MediaPipe)                                    │
│    └─ Full 640×480 frame → face_landmarker.detect()             │
│    └─ Returns: 468 landmarks, 51 blendshapes, 4×4 transform mat │
│    └─ Compute: iris coords, gaze ratio, blink scores, head pose │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. HAND DETECTION (MediaPipe)                                    │
│    └─ Full 640×480 frame → hand_landmarker.detect()             │
│    └─ Returns: 21 landmarks for first hand only                 │
│    └─ Compute: finger count, pinch, scroll, swipe               │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. GAZE PROCESSING                                               │
│    └─ Extract gaze_ratio (iris relative to eye socket)          │
│    └─ Extract head_pose (yaw, pitch from matrix)                │
│    └─ Apply head compensation: comp = gaze - head_pose×scale    │
│    └─ Apply homography transform (calibration)                  │
│    └─ Apply One-Euro smoothing                                  │
│    └─ Clamp to screen bounds [5, screen_w-5]                   │
│    └─ pyautogui.moveTo(x, y)                                    │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. GESTURE PROCESSING                                            │
│    └─ Wink detection → pyautogui.click()                        │
│    └─ Hand classification (finger_count → gesture type)         │
│    ├─ Scroll → pyautogui.scroll()                               │
│    ├─ Zoom → pyautogui.hotkey(mod, scroll)                      │
│    ├─ Drag → pyautogui.mouseDown()                              │
│    ├─ Switch Desktop → pyautogui.hotkey()                       │
│    └─ Mission Control → pyautogui.hotkey()                      │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. DRIFT CORRECTION (every 30s)                                  │
│    └─ Assume user is looking at screen center                   │
│    └─ Compute offset: (expected - actual)                       │
│    └─ Adjust homography translation: H[0,2] += dx, H[1,2] += dy │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. OVERLAY & DISPLAY                                             │
│    └─ Draw iris position (green dot)                            │
│    └─ Draw mouse coordinates                                    │
│    └─ Draw FPS counter                                          │
│    └─ Draw gesture/action text                                  │
│    └─ cv.imshow() to window                                     │
│    └─ cv.waitKey(1) for ESC exit                                │
└─────────────────────────────────────────────────────────────────┘
    ↓
FPS = 25–30 (limited by MediaPipe inference time ~30–40ms)
```

---

## Data Flow Summary

**Input:** 640×480 BGR frame from webcam @ ~30 FPS

**Through:**
1. MediaPipe Face (468 landmarks)
2. MediaPipe Hand (21 landmarks)
3. Geometric feature extraction (iris, gaze, blink, finger count)
4. One-Euro smoothing
5. Homography calibration
6. Gesture classification

**Output:** PyAutoGUI actions (moveTo, click, scroll, hotkey)

---

## Subsystem Dependencies

```
main.py
  ├─ VisionTracker (tracker.py)
  │   └─ MediaPipe (external, models in repo root)
  ├─ GazeProcessor (processor.py)
  │   └─ One-Euro filter implementation
  ├─ GazeCalibrator (calibration.py)
  │   └─ OpenCV (homography, perspective transform)
  ├─ GestureController (gestures.py)
  │   └─ Pure Python (no external CV)
  └─ PyAutoGUI
      └─ System-level mouse/keyboard control
```

---

## Configuration Parameters (All in main.py)

| Parameter | Value | Tunable? | Impact |
|-----------|-------|----------|--------|
| CAM_WIDTH | 640 | Yes | Higher = more detail, slower inference |
| CAM_HEIGHT | 480 | Yes | Higher = more detail, slower inference |
| HEAD_COMP_SCALE | 0.012 | Yes | Head compensation strength |
| SCROLL_SENSITIVITY | 15 | Yes | Scroll clicks per gesture frame |
| DRIFT_INTERVAL_S | 30 | Yes | Frequency of auto-drift correction |
| min_cutoff (One-Euro) | 0.8 | Yes | Lower = smoother, more lag |
| beta (One-Euro) | 0.02 | Yes | Higher = less lag, more jitter |
| wink_threshold | 0.06 | Yes | Minimum blink differential for wink |
| cooldown_ms (wink) | 500 | Yes | Minimum ms between wink triggers |

---

## Missing Observability

The system has **zero built-in logging or metrics**:

- No per-stage timing (detection, smoothing, display)
- No confidence score logging
- No gesture trigger counts or false-positive tracking
- No gaze accuracy measurements
- No FPS per subsystem
- No detection/tracking failure rates
- No profiling output (CPU time per model)

**Impact:** Impossible to diagnose bottlenecks or validate improvements without external profiling tools.

---

## Code Quality Notes

**Strengths:**
- Clean separation of concerns (tracker, processor, gestures, calibration)
- Good use of static methods and pure functions
- One-Euro filter is correctly implemented
- Comprehensive unit tests (59 tests)
- Proper error handling for missing models

**Weaknesses:**
- No type hints (Python 3.10+ syntax available but not used)
- No docstring for main.py
- Inconsistent hand detection logic (left/right not distinguished)
- No logging or observability
- Temporal state only tracked in gesture cooldowns (not in hand state machine)
- No abstraction for "detection + confidence gating"
