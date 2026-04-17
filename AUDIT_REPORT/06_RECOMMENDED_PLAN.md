# Recommended Implementation Plan

---

## Overview

This plan prioritizes improvements to maximize **accuracy, robustness, and FPS** within realistic time constraints. It balances quick wins (addressing highest-impact issues) with medium-term architectural improvements.

---

## Phase 1: Immediate Fixes This Week (7–10 hours)

**Goal:** Stabilize gesture recognition and improve robustness. No FPS gains yet.

### 1.1 Add Confidence Gating on Detections [1–2 hours]

**What:** Gate gaze and hand detections on MediaPipe confidence scores.

**Implementation:**
- Modify `tracker.py`: Add `get_gaze_ratio_with_confidence_gate()` (threshold 0.7)
- Modify `tracker.py`: Add `get_hand_landmarks_with_confidence_gate()` (threshold 0.6)
- Modify `main.py`: Use gated versions instead of raw getters
- Skip frames where confidence is too low (gaze becomes None, hand becomes None)

**Expected Benefit:**
- Eliminate ~50% of jitter from low-confidence frames
- Prevent false gestures during occlusions
- Accuracy: ±50px → ±40px

**Testing:**
- Unit test: Confidence gates reject <0.7 scores
- Integration test: Run at night (low light); verify graceful degradation

**Files Changed:**
- `src/core/tracker.py` (+20 lines)
- `src/main.py` (+5 lines)

**Risk:** LOW (can adjust threshold if too conservative)

---

### 1.2 Temporal Gesture Filtering [2–3 hours]

**What:** Add majority voting over 5-frame window before firing gesture.

**Implementation:**
- Modify `gestures.py`: Add `classify_hand_gesture_with_voting()`
- Maintain circular buffer of last 5 gesture classifications
- Only fire when buffer is ≥4/5 unanimous
- Clear buffer on hand loss (confidence drops)

**Example Code:**
```python
class GestureControllerWithVoting(GestureController):
    def __init__(self, *args, vote_window=5, **kwargs):
        super().__init__(*args, **kwargs)
        self._gesture_buffer = deque(maxlen=vote_window)
        self._vote_threshold = 4  # 4/5 frames
    
    def classify_hand_gesture_with_voting(self, hand_landmarks, finger_count):
        gesture = self.classify_hand_gesture(hand_landmarks, finger_count)
        
        if gesture is None:
            self._gesture_buffer.clear()
            return None
        
        self._gesture_buffer.append(gesture)
        
        if len(self._gesture_buffer) == self._gesture_buffer.maxlen:
            votes = Counter(self._gesture_buffer)
            stable_gesture, count = votes.most_common(1)[0]
            if count >= self._vote_threshold:
                return stable_gesture
        
        return None
```

**Expected Benefit:**
- Eliminate rapid gesture switching (zoom↔scroll↔zoom)
- Reduce false triggers by 80%
- False gesture rate: 30–40% → 5–10%

**Trade-off:** Adds ~150–200ms latency before gesture fires (acceptable for interface control)

**Testing:**
- Unit test: Verify voting logic (unanimous, majority, minority cases)
- Integration test: Hold 1 finger; verify scroll only fires once and sustains
- Integration test: Rapid 1→2→1 finger motion; verify no spurious zoom

**Files Changed:**
- `src/gestures/gestures.py` (+30 lines)
- `src/main.py` (+3 lines to call new method)

**Risk:** LOW (voting is a well-understood pattern)

---

### 1.3 Improved Finger Counting with Normalization [2–3 hours]

**What:** Normalize hand landmarks to canonical coordinate space before finger counting.

**Implementation:**
- Modify `tracker.py`: Add `normalize_hand_landmarks(landmarks)`
- Compute hand bounding box, normalize to [0, 1] range
- Modify `count_extended_fingers()` to use normalized landmarks
- Use Euclidean distance instead of single-axis comparisons

**Example Code:**
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

@staticmethod
def count_extended_fingers_improved(landmarks):
    """Count with normalization and confidence."""
    if not landmarks:
        return 0
    
    norm_lm = VisionTracker.normalize_hand_landmarks(landmarks)
    tip_mcp_pairs = [
        (4, 1),   # Thumb
        (8, 5),   # Index
        (12, 9),  # Middle
        (16, 13), # Ring
        (20, 17)  # Pinky
    ]
    
    count = 0
    for tip_idx, mcp_idx in tip_mcp_pairs:
        tip = norm_lm[tip_idx]
        mcp = norm_lm[mcp_idx]
        
        # Euclidean distance
        dist = math.sqrt((tip.x - mcp.x)**2 + (tip.y - mcp.y)**2)
        
        # Distance threshold (tuned empirically)
        threshold = 0.15 if tip_idx == 4 else 0.18
        
        if dist > threshold:
            count += 1
    
    return count
```

**Expected Benefit:**
- Remove camera-distance and hand-angle dependence
- Finger counting accuracy: ~75% → ~90%
- Gesture misclassification reduced by 15–20%

**Testing:**
- Unit test: Same hand at different distances; verify consistent finger count
- Unit test: Rotated hand (±30°); verify consistent counting
- Integration test: Hold 2-finger pinch at arm's length; verify zoom fires correctly

**Files Changed:**
- `src/core/tracker.py` (+30 lines)

**Risk:** LOW (normalization is robust; threshold can be tuned)

---

### 1.4 Add Timing Instrumentation [1–2 hours]

**What:** Add per-stage timing to identify real bottlenecks.

**Implementation:**
- Create `PerformanceMonitor` class
- Instrument each major stage: face detection, hand detection, gaze processing, gesture, display
- Print report every 10 seconds (or on-demand)

**Example Code:**
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
        print("\n--- Performance Report ---")
        for stage, durations in self.stages.items():
            avg = sum(durations) / len(durations)
            p99 = sorted(durations)[int(0.99 * len(durations))]
            print(f"{stage:20s}: avg={avg:6.2f}ms, p99={p99:6.2f}ms")

# Usage in main.py:
monitor = PerformanceMonitor()

while cap.isOpened():
    t0 = time.time()
    face_res, _ = tracker.process(frame)
    monitor.record("face_detect", (time.time() - t0) * 1000)
    
    t0 = time.time()
    _, hand_res = tracker.process(frame)
    monitor.record("hand_detect", (time.time() - t0) * 1000)
    
    ...
    
    if frame_count % 300 == 0:
        monitor.report()
```

**Expected Benefit:**
- Reveals which stage is actually slowest
- Informs Phase 2 prioritization
- No performance gain (diagnostic only)

**Testing:**
- Verify timing accuracy (compare to external profiler)

**Files Changed:**
- `src/main.py` (+50 lines, can be extracted to separate module)

**Risk:** LOW (diagnostic; no behavior change)

---

## Phase 1 Summary

| Task | Effort | Benefit | Risk |
|------|--------|---------|------|
| 1.1 Confidence gating | 1–2h | Jitter reduction, robustness | LOW |
| 1.2 Temporal filtering | 2–3h | 80% fewer false gestures | LOW |
| 1.3 Finger counting | 2–3h | 15% fewer misclassifications | LOW |
| 1.4 Profiling | 1–2h | Diagnostic insight | LOW |
| **Total** | **7–10h** | **Massive stability gain** | **LOW** |

**Expected Outcome:** System is now stable; gestures are reliable; gaze is smoother.

---

## Phase 2: Medium-Term Improvements (Week 2–3, 15–18 hours)

**Goal:** Increase FPS and robustness. Address head-pose and low-light issues.

### 2.1 ROI-Based Detection (Hand) [6–8 hours]

**What:** Run hand detection only every 5 frames. Predict hand position between detections.

**Implementation:**
1. Create `DetectionScheduler` class
2. Compute hand ROI from previous detection
3. Predict hand centroid using velocity model
4. Run full hand detection every 5 frames; track in between

**Benefit:**
- Hand detection cost drops from 25ms every frame to ~5ms average
- FPS improvement: 25–30 → 40–45 FPS (biggest single win)

**Trade-off:** Hand gestures lag slightly when hand moves quickly (acceptable for UI)

**Testing:**
- Verify hand tracking doesn't lose hand during rapid motion
- Check that ROI is wide enough (~100px margin)
- Test gesture recognition still works with tracked hand

**Files Changed:**
- `src/core/tracker.py` (+50 lines)
- `src/main.py` (+30 lines, new scheduling logic)

**Risk:** MEDIUM (prediction drift can cause hand loss)

---

### 2.2 Kalman Filtering for Gaze [3–4 hours]

**What:** Replace One-Euro filter with Kalman filter for more robust, principled smoothing.

**Implementation:**
1. Create `GazeKalmanFilter` class (similar to One-Euro, but Bayesian)
2. Track state: [x, y, vx, vy]
3. Weight updates by detection confidence
4. Replace call in `main.py`

**Benefit:**
- Smoother gaze under low light
- Graceful degradation during occlusions (predicts during gaps)
- Better handling of variable frame rates

**Trade-off:** Adds ~2–3ms latency (acceptable for cursor control)

**Testing:**
- Compare to One-Euro: should be visually similar
- Low-light test: Verify smoothness improves
- Occlusion test: Verify prediction bridges gaps

**Files Changed:**
- `src/core/processor.py` (+40 lines, new Kalman implementation)
- `src/main.py` (+5 lines, instantiation)

**Risk:** LOW (Kalman is well-understood; can tune Q, R)

---

### 2.3 Improved Calibration [2–3 hours]

**What:** Add stability checks during calibration; capture median gaze over 300ms window.

**Implementation:**
1. During calibration, accumulate gaze samples in a window
2. Display stability indicator (show when gaze is stable)
3. On SPACE press, use median gaze (not latest)
4. Log per-point confidence

**Benefit:**
- Reduce calibration errors from user movement (blinks, micro-movements)
- Gaze accuracy: ±50px → ±35px (post-calibration)

**Testing:**
- Calibrate while moving (should prompt for stability)
- Calibrate while blinking (should reject blink frames)
- Verify improved accuracy afterward

**Files Changed:**
- `src/main.py` (+40 lines, improved calibration loop)

**Risk:** LOW (improves existing calibration; backward compatible)

---

### 2.4 Add Recalibration Hotkey [2–3 hours]

**What:** Allow user to press 'R' during runtime to trigger fast 2-point recalibration.

**Implementation:**
1. Add 'R' key handler in main loop
2. Launch fast calibration (2 points: top-left and bottom-right)
3. Blend new homography with old (50% mix)

**Benefit:**
- Correct drift without restarting
- Support mid-session lighting changes

**Testing:**
- Press 'R', recalibrate; verify gaze accuracy improves
- Test blending (should be smooth transition)

**Files Changed:**
- `src/main.py` (+30 lines)
- `src/core/calibration.py` (+10 lines)

**Risk:** LOW (feature addition; no breaking changes)

---

### 2.5 Support 3D Head Pose (Optional, 6–8 hours)

**What:** Extract full 3D pose (yaw, pitch, roll) using solvePnP. Apply proper 3D rotation compensation.

**Implementation:**
1. Extract 6 face landmarks with known 3D positions
2. Use `cv.solvePnP()` to estimate 3D pose
3. Convert rotation vector to Euler angles
4. Apply 3D rotation compensation (not just linear scaling)

**Benefit:**
- Support head angles up to ±60° (vs ±30° currently)
- More principled compensation (works at any angle)

**Trade-off:** Requires camera intrinsic calibration (can use defaults)

**Testing:**
- Calibrate at center; test gaze tracking while turning head ±45°
- Compare 3D vs linear compensation

**Files Changed:**
- `src/core/tracker.py` (+40 lines)
- `src/main.py` (+10 lines)

**Risk:** MEDIUM (requires camera calibration; solvePnP can be unstable)

**Priority:** LOWER (only needed for head-mobile users)

---

## Phase 2 Summary

| Task | Effort | FPS Gain | Accuracy Gain | Risk |
|------|--------|----------|---------------|------|
| 2.1 ROI detection | 6–8h | +15 FPS | None | MEDIUM |
| 2.2 Kalman filter | 3–4h | +0 | +10% | LOW |
| 2.3 Improved calibration | 2–3h | +0 | +15% | LOW |
| 2.4 Recalibration hotkey | 2–3h | +0 | Dynamic | LOW |
| 2.5 3D head pose | 6–8h (optional) | +0 | +20% | MEDIUM |
| **Total** | **19–22h** | **+15 FPS** | **+25–45%** | **LOW–MEDIUM** |

**Expected Outcome:** System runs at 40–50 FPS; gaze is robust up to ±45° head angles; accurate calibration.

---

## Phase 3: Long-Term Improvements (Week 4+, Conditional)

**Goal:** Handle edge cases (low light, extreme angles, difficult lighting).

### 3.1 LSTM Gesture Recognizer [12–16 hours]

**When:** If gesture accuracy plateaus after Phases 1–2

**What:** Train LSTM on 10-frame sequences of hand landmarks.

**Expected Benefit:** 95%+ gesture accuracy (vs ~85% after Phase 1–2)

**Implementation Effort:** Data collection (100 samples per gesture), training, deployment

**Risk:** MEDIUM (requires training infrastructure; risk of overfitting)

---

### 3.2 Direct Pupil Detection with Fallback [8–12 hours]

**When:** If low-light tracking is critical use case

**What:** Add ellipse-fitting pupil detector as fallback to MediaPipe iris landmarks.

**Expected Benefit:** Works in low light, backlit, with glasses

**Risk:** MEDIUM (requires parameter tuning per camera)

---

### 3.3 GPU Acceleration [8–12 hours]

**When:** If FPS is still insufficient or CPU usage is critical

**What:** Move MediaPipe inference to GPU via ONNX Runtime or TensorRT

**Expected Benefit:** 40–50 FPS → 60–90 FPS

**Risk:** HIGH (requires GPU infrastructure, CUDA, driver compatibility)

---

## Decision Tree: Which Tasks to Do First?

**Start with Phase 1.** All 4 tasks are LOW-risk and have high impact (stability + diagnostics).

**After Phase 1, measure.**
- If FPS is <35, prioritize 2.1 (ROI detection) → +15 FPS
- If gaze is still jittery, prioritize 2.2 (Kalman) → smoother gaze
- If head movement is common use case, prioritize 2.5 (3D head pose)

**Typical order:**
1. Phase 1 (all 4 tasks) — 7–10h
2. Phase 2.1 (ROI detection) — 6–8h → +15 FPS
3. Phase 2.2 (Kalman) — 3–4h → robustness
4. Phase 2.3 (calibration) — 2–3h → accuracy
5. Phase 2.4 (recalibration) — 2–3h → UX
6. Phase 2.5 (3D head pose) — Optional, if needed

**Total for Phases 1–2.4:** 20–28 hours over 3 weeks
**Estimated FPS:** 25–30 → 40–50 FPS (80% improvement)
**Estimated Accuracy:** 30–40% false gestures → 5–10%

---

## Success Metrics

After Phase 1:
- [ ] Confidence gating passes unit tests
- [ ] Temporal filtering passes voting tests
- [ ] Finger counting passes normalization tests
- [ ] Profiling reveals actual bottleneck

After Phase 2:
- [ ] FPS ≥ 40 (measured via profiling)
- [ ] Gesture false-positive rate < 10% (manual testing)
- [ ] Gaze smooth under normal lighting (visual inspection)
- [ ] Calibration stable across 30 minutes of use

---

## Timeline

| Week | Phase | Tasks | Hours | Expected Outcome |
|------|-------|-------|-------|------------------|
| 1 | 1 | All (1.1–1.4) | 7–10 | Stable, diagnostic system |
| 2 | 2 | 2.1, 2.2, 2.3 | 11–14 | 40–50 FPS, robust gaze |
| 3 | 2 | 2.4, 2.5 (optional) | 4–11 | UX polish, edge cases |
| 4+ | 3 | LSTM, pupil, GPU (optional) | 28–40 | Production-ready (optional) |

**Realistic Target:** End of Week 3 = 40–50 FPS, <10% false gestures, robust tracking
**Stretch Goal:** End of Week 4 = 60–90 FPS, <2% false gestures, low-light support

