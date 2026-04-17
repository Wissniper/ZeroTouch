# Top 10 Action Items (Prioritized)

Each action item specifies:
- **Target:** Exact file(s) and lines
- **What:** Specific code/change needed
- **Why:** Which problem it solves
- **Effort:** Estimated hours
- **Benefit:** Expected improvement
- **Dependencies:** What must be done first
- **Testing:** How to validate

---

## 1. Add Confidence Gating on Gaze Detection [PHASE 1]

**Priority:** CRITICAL (Problem 6)  
**Effort:** 1–2 hours  
**Benefit:** Eliminate 50% of jitter; prevent false gaze jumps  
**Risk:** LOW

**Target:**
- `src/core/tracker.py`, lines 80–107 (get_gaze_ratio function)
- `src/main.py`, lines 169, 172

**What to Do:**
1. In `tracker.py`, create new function:
```python
def get_gaze_ratio_with_confidence_gate(self, face_result, min_confidence=0.7):
    """Return gaze ratio only if iris/eye landmarks are confident."""
    if not face_result.face_landmarks:
        return None
    
    marks = face_result.face_landmarks[0]
    iris_indices = list(range(468, 478))
    iris_confs = [marks[i].z for i in iris_indices]
    avg_iris_conf = sum(iris_confs) / len(iris_confs)
    
    # Check eye socket confidence too
    eye_indices = [33, 133, 159, 145, 362, 263, 386, 374]
    eye_confs = [marks[i].z for i in eye_indices]
    avg_eye_conf = sum(eye_confs) / len(eye_confs)
    
    if avg_iris_conf < min_confidence or avg_eye_conf < min_confidence:
        return None
    
    # Call existing get_gaze_ratio logic
    return self.get_gaze_ratio(face_result)
```

2. In `main.py`, replace line 169:
```python
# OLD:
gaze_ratio = tracker.get_gaze_ratio(face_res)

# NEW:
gaze_ratio = tracker.get_gaze_ratio_with_confidence_gate(face_res, min_confidence=0.7)
```

3. Update condition at line 172 to handle None:
```python
if gaze_ratio is None:
    continue  # Skip this frame; confidence too low
```

**Why:** MediaPipe returns confidence per landmark; ignoring it causes false jumps during occlusions/blinks

**Testing:**
```python
# Unit test: test_gaze_confidence_gating
face_res_high_conf = ...  # Create mock with z=0.9
face_res_low_conf = ...   # Create mock with z=0.3
assert tracker.get_gaze_ratio_with_confidence_gate(face_res_high_conf, 0.7) is not None
assert tracker.get_gaze_ratio_with_confidence_gate(face_res_low_conf, 0.7) is None

# Integration test: dim lighting
# Run system in low light; verify cursor doesn't jump
```

**Dependencies:** None (standalone enhancement)

**Success Criteria:**
- Gaze is None on ~5% of frames (low-light/occlusion)
- Cursor movement is smoother (reduced jitter)
- No false gaze jumps during blinks

---

## 2. Implement Temporal Gesture Voting [PHASE 1]

**Priority:** CRITICAL (Problem 1)  
**Effort:** 2–3 hours  
**Benefit:** Eliminate 80% of false gesture triggers  
**Risk:** LOW

**Target:**
- `src/gestures/gestures.py`, line 169 (classify_hand_gesture method)
- `src/main.py`, line 211

**What to Do:**
1. In `gestures.py`, add to `__init__()`:
```python
from collections import deque, Counter

def __init__(self, ...):
    ...
    self._gesture_vote_buffer = deque(maxlen=5)
    self._vote_threshold = 4  # 4/5 frames unanimous
```

2. Add new method to `GestureController`:
```python
def classify_hand_gesture_with_voting(self, hand_landmarks, finger_count):
    """Classify gesture with temporal voting."""
    gesture = self.classify_hand_gesture(hand_landmarks, finger_count)
    
    if gesture is None:
        self._gesture_vote_buffer.clear()
        return None
    
    self._gesture_vote_buffer.append(gesture)
    
    if len(self._gesture_vote_buffer) == self._gesture_vote_buffer.maxlen:
        votes = Counter(self._gesture_vote_buffer)
        stable_gesture, count = votes.most_common(1)[0]
        if count >= self._vote_threshold:
            return stable_gesture
    
    return None
```

3. In `main.py`, replace line 211:
```python
# OLD:
gesture_name = gestures.classify_hand_gesture(hand_lm, finger_count)

# NEW:
gesture_name = gestures.classify_hand_gesture_with_voting(hand_lm, finger_count)
```

**Why:** Frame-by-frame classification fires transient gestures (1→2→1 finger causes multiple action triggers)

**Testing:**
```python
# Unit test: test_gesture_voting
gc = GestureController()
assert gc.classify_hand_gesture_with_voting(hand, 1) is None  # First frame
for i in range(4):
    assert gc.classify_hand_gesture_with_voting(hand, 1) is None  # Frames 2–5
assert gc.classify_hand_gesture_with_voting(hand, 1) == "scroll"  # Frame 6

# Test mode switching:
gc = GestureController()
for i in range(4):
    gc.classify_hand_gesture_with_voting(hand, 1)
result = gc.classify_hand_gesture_with_voting(hand, 2)  # Different gesture
assert result is None  # No spurious "zoom" on first frame of new gesture

# Integration test: rapid pinch-release
# Move fingers 1→2→1 in 200ms; verify no scroll+zoom spam
```

**Dependencies:** None

**Success Criteria:**
- Single gesture sustained for 150ms+ fires reliably
- Transient gesture changes (1→2→1) don't fire spurious actions
- Gesture response latency <200ms (acceptable)

---

## 3. Normalize Hand Landmarks for Finger Counting [PHASE 1]

**Priority:** HIGH (Problem 2)  
**Effort:** 2–3 hours  
**Benefit:** 15–20% fewer gesture misclassifications  
**Risk:** LOW

**Target:**
- `src/core/tracker.py`, lines 145–174 (count_extended_fingers method)

**What to Do:**
1. Add normalize function to `VisionTracker`:
```python
@staticmethod
def normalize_hand_landmarks(landmarks):
    """Normalize landmarks to [0,1] relative to hand bounding box."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    
    normalized = []
    for lm in landmarks:
        n = SimpleNamespace(
            x=(lm.x - min_x) / width,
            y=(lm.y - min_y) / height,
            z=lm.z
        )
        normalized.append(n)
    return normalized
```

2. Replace `count_extended_fingers()`:
```python
@staticmethod
def count_extended_fingers(landmarks):
    """Count extended fingers with normalization."""
    if landmarks is None:
        return 0
    
    norm_lm = VisionTracker.normalize_hand_landmarks(landmarks)
    tip_mcp = [(4,1), (8,5), (12,9), (16,13), (20,17)]
    
    count = 0
    for tip_idx, mcp_idx in tip_mcp:
        tip, mcp = norm_lm[tip_idx], norm_lm[mcp_idx]
        dist = math.sqrt((tip.x - mcp.x)**2 + (tip.y - mcp.y)**2)
        threshold = 0.15 if tip_idx == 4 else 0.18
        if dist > threshold:
            count += 1
    
    return count
```

**Why:** Current heuristic fails for rotated hands and different camera distances

**Testing:**
```python
# Unit test: test_finger_normalization
hand_near = ...  # Hand 30cm from camera
hand_far = ...   # Same hand 60cm away
assert count_extended_fingers(hand_near) == count_extended_fingers(hand_far)

# Unit test: test_rotated_hand
hand_straight = ...
hand_rotated_30deg = ...
assert count_extended_fingers(hand_straight) == count_extended_fingers(hand_rotated_30deg)

# Integration test: pinch at arm's length
# Verify zoom fires (not misclassified as scroll)
```

**Dependencies:** None

**Success Criteria:**
- Finger count is consistent regardless of hand distance/angle
- Gesture misclassification rate <10% (from ~25%)

---

## 4. Add Timing Instrumentation [PHASE 1]

**Priority:** HIGH (Problem 8)  
**Effort:** 1–2 hours  
**Benefit:** Diagnostic (identifies real bottleneck)  
**Risk:** LOW

**Target:**
- `src/main.py`, main loop (lines 151–290)

**What to Do:**
1. Add to imports:
```python
from collections import deque
import time
```

2. Before main loop, add:
```python
class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.stages = {}
        self.window_size = window_size
    
    def record(self, stage_name, duration_ms):
        if stage_name not in self.stages:
            self.stages[stage_name] = deque(maxlen=self.window_size)
        self.stages[stage_name].append(duration_ms)
    
    def report(self):
        print("\n--- Performance Report (last {} frames) ---".format(self.window_size))
        for stage, durations in sorted(self.stages.items()):
            avg = sum(durations) / len(durations)
            max_d = max(durations)
            print(f"{stage:20s}: avg={avg:6.2f}ms, max={max_d:6.2f}ms")

monitor = PerformanceMonitor()
```

3. Instrument main loop:
```python
while cap.isOpened():
    t_frame_start = time.time()
    ret, frame = cap.read()
    
    # FACE DETECTION
    t0 = time.time()
    face_res, _ = tracker.process(frame)
    monitor.record("face_detect", (time.time() - t0) * 1000)
    
    # HAND DETECTION
    t0 = time.time()
    _, hand_res = tracker.process(frame)  # TODO: combine with face
    monitor.record("hand_detect", (time.time() - t0) * 1000)
    
    # GAZE PROCESSING
    t0 = time.time()
    gaze_ratio = tracker.get_gaze_ratio(face_res)
    head_pose = tracker.get_head_pose(face_res)
    # ... compensation, transform, smoothing ...
    monitor.record("gaze_process", (time.time() - t0) * 1000)
    
    # GESTURE PROCESSING
    t0 = time.time()
    # ... all gesture logic ...
    monitor.record("gesture_process", (time.time() - t0) * 1000)
    
    # DISPLAY
    t0 = time.time()
    cv.imshow("IrisFlow Feed", frame)
    cv.waitKey(1)
    monitor.record("display", (time.time() - t0) * 1000)
    
    # TOTAL FRAME
    frame_time = (time.time() - t_frame_start) * 1000
    fps = 1000.0 / frame_time if frame_time > 0 else 0
    
    if frame_count % 300 == 0:
        monitor.report()
        print(f"FPS: {fps:.1f}")
    
    frame_count += 1
```

**Why:** Can't optimize without knowing which stage is slowest

**Testing:**
```bash
# Run system normally and observe output every 10 seconds
# Expected output:
# face_detect        : avg= 35.2ms, max= 45.3ms
# hand_detect        : avg= 25.1ms, max= 32.1ms
# gaze_process       : avg=  2.3ms, max=  5.2ms
# gesture_process    : avg=  1.8ms, max=  3.1ms
# display            : avg= 10.5ms, max= 15.2ms
```

**Dependencies:** None

**Success Criteria:**
- Monitor reports all stages
- Total time per frame is ~80–100ms (10–12.5 FPS expected before optimization)

---

## 5. ROI-Based Hand Detection [PHASE 2]

**Priority:** CRITICAL (Problem 5, biggest FPS gain)  
**Effort:** 6–8 hours  
**Benefit:** +15–20 FPS (25–30 → 40–50 FPS)  
**Risk:** MEDIUM

**Target:**
- `src/core/tracker.py` (new method)
- `src/main.py`, line 167 (detection call)

**What to Do:**
1. In `tracker.py`, add ROI extraction:
```python
def get_hand_roi(self, hand_landmarks, frame_shape, margin_ratio=0.3):
    """Get bounding box around detected hand."""
    if hand_landmarks is None:
        return None
    
    h, w = frame_shape[:2]
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width = (max_x - min_x) * w * (1 + margin_ratio)
    height = (max_y - min_y) * h * (1 + margin_ratio)
    
    center_x = (min_x + max_x) * w / 2
    center_y = (min_y + max_y) * h / 2
    
    x1 = int(max(0, center_x - width / 2))
    y1 = int(max(0, center_y - height / 2))
    x2 = int(min(w, center_x + width / 2))
    y2 = int(min(h, center_y + height / 2))
    
    return (x1, y1, x2, y2)

def detect_hand_in_roi(self, frame, roi):
    """Run hand detection on cropped ROI."""
    if roi is None:
        # Fall back to full-frame
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.hand_detector.detect(mp_image)
    
    x1, y1, x2, y2 = roi
    roi_frame = frame[y1:y2, x1:x2]
    
    if roi_frame.shape[0] < 20 or roi_frame.shape[1] < 20:
        # ROI too small; do full-frame
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.hand_detector.detect(mp_image)
    
    rgb_roi = cv.cvtColor(roi_frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_roi)
    result = self.hand_detector.detect(mp_image)
    
    # Adjust landmark coordinates back to full frame
    if result.hand_landmarks:
        for hand_list in result.hand_landmarks:
            for lm in hand_list:
                lm.x = (lm.x * roi_frame.shape[1] + x1) / frame.shape[1]
                lm.y = (lm.y * roi_frame.shape[0] + y1) / frame.shape[0]
    
    return result
```

2. In `main.py`, add scheduler:
```python
class HandDetectionScheduler:
    def __init__(self, interval=5):
        self.interval = interval
        self.frame_count = 0
        self.last_hand_roi = None
    
    def should_detect(self):
        return self.frame_count % self.interval == 0
    
    def update(self, hand_landmarks, frame_shape):
        roi = tracker.get_hand_roi(hand_landmarks, frame_shape)
        self.last_hand_roi = roi
        self.frame_count += 1
    
    def get_roi(self):
        return self.last_hand_roi

scheduler = HandDetectionScheduler(interval=5)
```

3. In main loop, replace line 167:
```python
# OLD:
face_res, hand_res = tracker.process(frame)

# NEW:
face_res, _ = tracker.process(frame)

if scheduler.should_detect():
    hand_res = tracker.detect_hand_in_roi(frame, scheduler.get_roi())
else:
    hand_res = tracker.detect_hand_in_roi(frame, scheduler.get_roi())

if hand_res.hand_landmarks:
    scheduler.update(hand_res.hand_landmarks[0], frame.shape)
```

**Why:** Hand detection costs 20–30ms; running only every 5 frames saves 80% of that cost

**Testing:**
```python
# Unit test: test_roi_extraction
hand_lm = ...
roi = tracker.get_hand_roi(hand_lm, (480, 640))
assert roi[0] < roi[2]  # x1 < x2
assert roi[1] < roi[3]  # y1 < y3

# Integration test: rapid hand motion
# Move hand quickly; verify tracking doesn't lose hand
# Verify gesture still fires correctly

# Performance test:
# Measure FPS before/after; expect +15–20 FPS
```

**Dependencies:** Problem 4 (profiling) to validate FPS improvement

**Success Criteria:**
- Hand detection costs drop to ~5ms per frame average (was 25ms)
- FPS improves to 40–50 (was 25–30)
- Hand gestures still work reliably

---

## 6. Replace One-Euro with Kalman Filter [PHASE 2]

**Priority:** HIGH (Problem 6)  
**Effort:** 3–4 hours  
**Benefit:** Smoother gaze, graceful degradation during occlusions  
**Risk:** LOW

**Target:**
- `src/core/processor.py` (new Kalman implementation)
- `src/main.py`, line 181 (processor call)

**What to Do:**
1. In `processor.py`, replace `_OneEuroFilter` with Kalman (or keep both):
```python
class GazeKalmanFilter:
    def __init__(self, process_variance=0.001, measurement_variance=0.01):
        self.x = 0.5  # state
        self.p = 0.1  # covariance
        self.q = process_variance  # process noise
        self.r = measurement_variance  # measurement noise
    
    def filter(self, z, confidence=1.0):
        """Update with measurement; weight by confidence."""
        # Predict
        self.p += self.q
        
        # Update (if confidence is high)
        if confidence < 0.5:
            # Low confidence; skip update (keep prediction)
            return self.x
        
        # Adaptive measurement noise (low confidence → high noise)
        r_adaptive = self.r / (confidence + 0.1)
        
        y = z - self.x
        s = self.p + r_adaptive
        k = self.p / s
        self.x = self.x + k * y
        self.p = (1 - k) * self.p
        
        return self.x

class GazeProcessor:
    def __init__(self, min_cutoff=0.5, beta=0.05, use_kalman=True):
        self.use_kalman = use_kalman
        if use_kalman:
            self._kx = GazeKalmanFilter()
            self._ky = GazeKalmanFilter()
        else:
            self._fx = _OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
            self._fy = _OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
    
    def process(self, raw_x, raw_y, confidence=1.0, t=None):
        """Return smoothed (x, y)."""
        if self.use_kalman:
            return self._kx.filter(raw_x, confidence), self._ky.filter(raw_y, confidence)
        else:
            return self._fx.filter(raw_x, t), self._fy.filter(raw_y, t)
```

2. In `main.py`, update processor instantiation:
```python
processor = GazeProcessor(use_kalman=True)
```

3. Update processor call to pass confidence:
```python
gaze_confidence = compute_gaze_confidence(face_res)
sx, sy = processor.process(raw_sx, raw_sy, confidence=gaze_confidence)
```

**Why:** Kalman is principled Bayesian filter; adapts to confidence; predicts during occlusions

**Testing:**
```python
# Unit test: test_kalman_convergence
kf = GazeKalmanFilter()
for i in range(10):
    result = kf.filter(0.5)
assert abs(result - 0.5) < 0.1  # Converges to measurement

# Unit test: test_confidence_weighting
kf = GazeKalmanFilter()
kf.filter(0.5, confidence=0.9)  # High confidence
kf.filter(0.1, confidence=0.2)  # Low confidence → should ignore
# Result should be closer to 0.5 than to 0.1

# Integration test: compare to One-Euro
# Run both Kalman and One-Euro on same gaze data
# Visual inspection: should be similar smoothness
```

**Dependencies:** Problem 4 (profiling) to measure impact

**Success Criteria:**
- Gaze smoothness is comparable or better than One-Euro
- Graceful degradation during low-confidence frames
- Prediction bridges brief occlusions (<200ms)

---

## 7. Improved Calibration with Stability Checks [PHASE 2]

**Priority:** MEDIUM (Problem 7)  
**Effort:** 2–3 hours  
**Benefit:** Calibration errors reduced; gaze accuracy +15%  
**Risk:** LOW

**Target:**
- `src/main.py`, run_calibration function (lines 43–108)

**What to Do:**
1. Modify calibration loop to accumulate gaze:
```python
gaze_history = deque(maxlen=10)

while point_idx < 9:
    ...
    face_res, _ = tracker.process(frame)
    gaze_ratio = tracker.get_gaze_ratio(face_res)
    gaze_confidence = compute_gaze_confidence(face_res)
    
    if gaze_ratio and gaze_confidence > 0.7:
        gaze_history.append(gaze_ratio)
    
    # Display stability indicator
    if len(gaze_history) >= 5:
        gaze_array = np.array(list(gaze_history))
        gaze_std = np.std(gaze_array, axis=0)
        stability = np.max(gaze_std)
        
        status_color = (0, 255, 0) if stability < 0.05 else (0, 255, 255)
        cv.putText(bg, f"Stability: {stability:.3f}", (20, 100),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    else:
        cv.putText(bg, "Waiting for stable gaze...", (20, 100),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    ...
    
    key = cv.waitKey(1) & 0xFF
    if key == 32:  # SPACE
        if len(gaze_history) < 5:
            print("  Need more stable samples; wait for green indicator")
            continue
        
        # Use median gaze (robust to transients)
        gaze_array = np.array(list(gaze_history))
        gaze_median = np.median(gaze_array, axis=0)
        
        calibrator.add_calibration_point(
            target_x, target_y, gaze_median[0], gaze_median[1]
        )
        print(f"  Captured {point_idx + 1}/9")
        point_idx += 1
        gaze_history.clear()
```

2. Add helper function:
```python
def compute_gaze_confidence(face_result):
    """Compute overall confidence of gaze estimate."""
    if not face_result.face_landmarks:
        return 0.0
    marks = face_result.face_landmarks[0]
    iris_indices = list(range(468, 478))
    iris_confs = [marks[i].z for i in iris_indices]
    return sum(iris_confs) / len(iris_confs)
```

**Why:** User movement/blinks during calibration tap cause bad calibration points

**Testing:**
```python
# Manual test: intentional movement during calibration
# User moves head during tap → system should prompt to wait
# User blinks during tap → median should ignore blink frame
# Verify calibration accuracy improves

# Measure: gaze error after calibration
# Should be <±30px (was ±50px)
```

**Dependencies:** Problem 1.1 (confidence gating, which computes gaze confidence)

**Success Criteria:**
- Stability indicator displays correctly
- User is prompted to wait if gaze is unstable
- Gaze accuracy post-calibration improves to ±35px

---

## 8. Add Recalibration Hotkey [PHASE 2]

**Priority:** MEDIUM (UX improvement)  
**Effort:** 2–3 hours  
**Benefit:** Dynamic drift correction without restart  
**Risk:** LOW

**Target:**
- `src/main.py`, main loop (lines 151–290)

**What to Do:**
1. In main loop, add recalibration check:
```python
key = cv.waitKey(1) & 0xFF
if key == 27:  # ESC
    break
elif key == ord('r'):  # 'R' for recalibration
    print("Starting fast recalibration (2 points)...")
    new_calibrator = run_fast_calibration(cap, tracker, screen_w, screen_h)
    if new_calibrator:
        # Blend new with old (50% each)
        if calibrator.transform_matrix is not None and new_calibrator.transform_matrix is not None:
            blend_factor = 0.5
            calibrator.transform_matrix = (
                blend_factor * new_calibrator.transform_matrix +
                (1 - blend_factor) * calibrator.transform_matrix
            )
        else:
            calibrator = new_calibrator
        print("Recalibration complete.")
```

2. Add fast calibration function:
```python
def run_fast_calibration(cap, tracker, screen_w, screen_h):
    """Fast 2-point recalibration (top-left and bottom-right)."""
    calibrator = GazeCalibrator()
    points = [(int(0.2 * screen_w), int(0.2 * screen_h)),
              (int(0.8 * screen_w), int(0.8 * screen_h))]
    
    for i, (target_x, target_y) in enumerate(points):
        print(f"Look at point {i+1}/2 and press SPACE")
        
        for _ in range(300):  # 10 second timeout
            ret, frame = cap.read()
            if not ret:
                break
            
            cv.flip(frame, 1, frame)
            bg = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            cv.circle(bg, (target_x, target_y), 15, (0, 0, 255), -1)
            cv.circle(bg, (target_x, target_y), 5, (255, 255, 255), -1)
            cv.imshow("Recalibration", bg)
            
            face_res, _ = tracker.process(frame)
            gaze_ratio = tracker.get_gaze_ratio(face_res)
            
            key = cv.waitKey(1) & 0xFF
            if key == 32 and gaze_ratio:  # SPACE
                calibrator.add_calibration_point(target_x, target_y, gaze_ratio[0], gaze_ratio[1])
                print(f"  Captured point {i+1}")
                break
    
    cv.destroyWindow("Recalibration")
    
    if calibrator.calculate_mapping():
        return calibrator
    return None
```

**Why:** Allows drift correction without restart; supports lighting/setup changes

**Testing:**
```python
# Manual test:
# 1. Calibrate normally
# 2. Wait 5 minutes; gaze drifts
# 3. Press 'R'; do 2-point recalibration
# 4. Verify gaze is re-centered

# Measure: gaze error before/after recalibration
# Should drop from drift level (~50px) back to baseline (~30px)
```

**Dependencies:** Problem 7 (improved calibration with stability checks)

**Success Criteria:**
- 'R' key triggers recalibration dialog
- Fast calibration completes in <1 minute
- Gaze accuracy restored after recalibration

---

## 9. Support 3D Head Pose Compensation [PHASE 2, OPTIONAL]

**Priority:** MEDIUM (only if head motion is common)  
**Effort:** 6–8 hours  
**Benefit:** Support head angles up to ±60° (vs ±30°)  
**Risk:** MEDIUM

**Target:**
- `src/core/tracker.py` (new 3D pose extraction)
- `src/main.py` (head compensation logic)

**What to Do:**
1. In `tracker.py`, add 3D head pose extraction:
```python
def get_head_pose_3d(self, face_result):
    """Extract full 3D head pose (yaw, pitch, roll) using solvePnP."""
    if not face_result.face_landmarks:
        return None
    
    marks = face_result.face_landmarks[0]
    
    # Known 3D positions of reference landmarks (rough estimates)
    ref_points_3d = np.array([
        marks[1].x, marks[1].y, marks[1].z,      # nose tip
        marks[152].x, marks[152].y, marks[152].z,  # chin
        marks[33].x, marks[33].y, marks[33].z,    # left eye
        marks[263].x, marks[263].y, marks[263].z, # right eye
    ], dtype=np.float32).reshape(4, 3)
    
    # Actual 3D reference points (from face model)
    object_points = np.array([
        [0, 0, 25],      # nose tip
        [0, -50, 0],     # chin
        [-30, 10, 20],   # left eye
        [30, 10, 20]     # right eye
    ], dtype=np.float32)
    
    # Project to pixel coordinates
    image_points = np.array([
        [marks[1].x * 640, marks[1].y * 480],
        [marks[152].x * 640, marks[152].y * 480],
        [marks[33].x * 640, marks[33].y * 480],
        [marks[263].x * 640, marks[263].y * 480]
    ], dtype=np.float32)
    
    # Camera matrix (assuming 640×480 camera)
    focal_length = 640
    camera_matrix = np.array([
        [focal_length, 0, 320],
        [0, focal_length, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros(4, dtype=np.float32)
    
    success, rot_vec, trans_vec = cv.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs
    )
    
    if not success:
        return None
    
    # Convert to Euler angles
    rot_mat, _ = cv.Rodrigues(rot_vec)
    yaw = np.arctan2(rot_mat[0, 2], rot_mat[2, 2])
    pitch = np.arcsin(-rot_mat[1, 2])
    roll = np.arctan2(rot_mat[1, 0], rot_mat[1, 1])
    
    return (yaw, pitch, roll)
```

2. In `main.py`, apply 3D compensation:
```python
head_pose_3d = tracker.get_head_pose_3d(face_res)

if gaze_ratio and head_pose_3d:
    yaw, pitch, roll = head_pose_3d
    
    # 3D compensation (proper rotation)
    gaze_3d = np.array([
        gaze_ratio[0] - 0.5,
        0.5 - gaze_ratio[1],
        1.0
    ])
    gaze_3d = gaze_3d / (np.linalg.norm(gaze_3d) + 1e-6)
    
    # Create rotation matrices
    R_yaw = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    
    # Apply rotation
    R = R_yaw @ R_pitch
    gaze_comp_3d = R.T @ gaze_3d
    
    comp_x = gaze_comp_3d[0] + 0.5
    comp_y = 0.5 - gaze_comp_3d[1]
    comp = (comp_x, comp_y)
else:
    comp = gaze_ratio  # Fallback
```

**Why:** Linear compensation fails for head angles >±30°

**Testing:**
```python
# Integration test: head rotation
# 1. Calibrate at center
# 2. Turn head 45° left while looking at fixed point
# 3. Verify cursor stays on that point (not head-compensated)
# 4. Compare 3D compensation vs linear (should be better)
```

**Dependencies:** Problem 4 (profiling to measure performance impact)

**Success Criteria:**
- 3D head pose extracted successfully
- Compensation works up to ±60° head angles
- Gaze accuracy improves for large head rotations

---

## 10. Prepare LSTM Gesture Model (Optional, PHASE 3)

**Priority:** LOWER (only if gesture accuracy plateaus)  
**Effort:** 12–16 hours  
**Benefit:** 95%+ gesture accuracy (vs ~85% after Phase 1–2)  
**Risk:** MEDIUM

**Target:**
- New file: `src/gestures/gesture_lstm.py`
- New directory: `models/gesture_lstm/` (model weights)

**What to Do:**
1. Collect training data (100–200 labeled examples per gesture)
   - Record video of each gesture type
   - Annotate frames with gesture label
   - Extract hand landmarks + finger count sequences

2. Train LSTM:
```python
import torch
from torch import nn

class GestureLSTM(nn.Module):
    def __init__(self, input_size=42, hidden_size=128, num_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

# Training loop (pseudo-code)
model = GestureLSTM()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

for epoch in range(100):
    for batch in train_loader:
        x, y = batch
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

3. Deploy in `main.py`:
```python
from src.gestures.gesture_lstm import GestureLSTMInference

gesture_lstm = GestureLSTMInference(model_path="models/gesture_lstm/model.pt")

# In main loop:
gesture_lstm.add_frame(hand_lm, finger_count)
gesture_name = gesture_lstm.predict()  # Returns prediction every 10 frames
```

**Why:** Frame-by-frame classification is fundamentally limited; LSTM learns temporal patterns

**Testing:**
```python
# Separate test set: run model on unseen gesture videos
# Measure accuracy: should be 95%+
# Measure latency: should be <30ms per frame (acceptable)
```

**Dependencies:** Problem 1.1–1.2 (temporal filtering foundation; LSTM is the upgrade path)

**Success Criteria:**
- Model achieves >95% accuracy on test set
- Inference latency <30ms per frame
- Generalizes across users (or requires per-user training)

---

## Timeline Summary

| Action | Phase | Week | Hours | Prerequisites |
|--------|-------|------|-------|------------------|
| 1. Confidence gating | 1 | 1 | 1–2 | None |
| 2. Temporal voting | 1 | 1 | 2–3 | None |
| 3. Finger normalization | 1 | 1 | 2–3 | None |
| 4. Timing instrumentation | 1 | 1 | 1–2 | None |
| 5. ROI hand detection | 2 | 2 | 6–8 | #4 |
| 6. Kalman gaze filter | 2 | 2 | 3–4 | #1 |
| 7. Improved calibration | 2 | 2 | 2–3 | #1 |
| 8. Recalibration hotkey | 2 | 3 | 2–3 | #7 |
| 9. 3D head pose | 2 | 3 | 6–8 | #4 (optional) |
| 10. LSTM gesture | 3 | 4 | 12–16 | #2 (optional) |

**Total:** ~40–50 hours over 4 weeks (or 22–28 hours for just Phases 1–2.4)

