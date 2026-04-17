# Appendix: Assumptions, Questions, and Metrics

---

## A. Key Assumptions Made During Audit

### System-Level Assumptions

1. **Webcam Quality:** 640×480 resolution is achievable; decent image quality (no severe chromatic aberration, distortion)
   - **If untrue:** Need to add undistortion or higher-res camera

2. **Processing Target:** Aiming for 30–60 FPS is acceptable (not real-time <5ms requirement)
   - **If untrue:** GPU acceleration becomes mandatory; CPU-only is insufficient

3. **Head Position:** User is seated ~60cm from camera, facing mostly forward
   - **If untrue:** Need to support wider range of head poses; 3D compensation becomes critical

4. **Lighting:** Normal indoor lighting (not pitch dark, not direct sunlight)
   - **If untrue:** Need to add glint suppression, adaptive exposure, or infrared IR LEDs

5. **Single User:** System is single-user (not multi-person in frame)
   - **If untrue:** Need to track multiple eyes; current MediaPipe settings (num_faces=1) would reject other faces

### Code-Level Assumptions

1. **MediaPipe Correctness:** MediaPipe blendshapes (indices 9, 10) correctly reflect eye closure
   - **If untrue:** Wink detection will fail; need custom model or manual landmarks

2. **Homography Linearity:** Iris-to-screen mapping is sufficiently linear (homography assumption is valid)
   - **If untrue:** Calibration errors will persist; need non-linear warping

3. **One-Euro Parameters:** min_cutoff=0.8, beta=0.02 are reasonable for typical users
   - **If untrue:** Need personalized tuning or adaptive parameter selection

4. **Finger Counting Thresholds:** Distance thresholds 0.15–0.18 (normalized coords) are correct
   - **If untrue:** Finger counts will be wrong; need re-tuning per camera

5. **Gesture Cooldown:** 500ms cooldown is sufficient to prevent accidental re-triggers
   - **If untrue:** Need shorter cooldown or state machine (Action Item #9)

### Use-Case Assumptions

1. **Cursor Control:** Primary use is cursor control, not precise gaze-based text input
   - **Impact:** ±30px error is acceptable; for text, need <10px

2. **Gesture Frequency:** Gestures are infrequent (not continuous; e.g., "scroll 10 times", then rest)
   - **If untrue:** Gesture spam prevention becomes critical; majority voting (2–3 hrs, Action Item #2) is essential

3. **No Glasses:** User may wear glasses, but not heavy reflective frames
   - **If untrue:** Glint suppression critical; may need IR-based tracking

4. **Gesture Familiarity:** User is familiar with hand gestures (pinch, swipe, etc.)
   - **If untrue:** Need gesture tutorialization or simpler gesture set

---

## B. Unanswered Questions (For User)

### 1. **Hardware Constraints**

- **Q:** Is GPU (CUDA) available? (Affects optimization path)
  - **If YES:** Can achieve 60–90 FPS; GPU acceleration (Phase 3) becomes worthwhile
  - **If NO:** Ceiling is ~40–50 FPS; skip GPU work

- **Q:** What's the target deployment environment? (Desktop, laptop, mobile, embedded?)
  - **Desktop/Laptop:** CPU-based is fine; 40–50 FPS is acceptable
  - **Mobile/Embedded:** Need aggressive optimization (GPU, quantization)

- **Q:** Is camera calibration data available (focal length, principal point)?
  - **If YES:** 3D head pose (Action Item #9) becomes more accurate
  - **If NO:** Use default values; expect some calibration error

### 2. **Use-Case & User Profile**

- **Q:** What's the primary use case? (Cursor control, text input, accessibility, gaming?)
  - **Cursor control:** Current system is fine; focus on stability (Phase 1)
  - **Text input:** Need ±10px gaze accuracy; Phase 2–3 essential
  - **Gaming:** Need 60+ FPS; GPU acceleration essential
  - **Accessibility (motor impairment):** Need high reliability; Phase 3 (LSTM) recommended

- **Q:** Will users have glasses or sunglasses? (Affects iris tracking robustness)
  - **Glasses common:** Glint suppression (Problem 3B) becomes important
  - **Sunglasses common:** May need infrared (significant redesign)

- **Q:** What head angles will users typically use? (±30°, ±60°, free range?)
  - **±30°:** Current linear compensation is fine
  - **±60°:** Need 3D head pose (Action Item #9)
  - **Free range:** Need full 3D + fallback trackers

- **Q:** Will lighting conditions vary significantly? (Office, outdoors, dim room?)
  - **Office (controlled):** Current system should work; Phase 1 stabilization sufficient
  - **Outdoors + indoors:** Need CLAHE + glint suppression (Problem 3B)
  - **Dim room:** May need infrared; current MediaPipe iris may fail

### 3. **Gesture & Feature Priorities**

- **Q:** Which gestures are most important? (Priority ranking: scroll, zoom, drag, swipe, palm?)
  - **Affects:** Phase 1 focus (temporal voting for most-used gesture first)

- **Q:** Do you need multi-hand support? (Currently only detects first hand)
  - **YES:** Significant refactoring; separate tracking per hand
  - **NO:** Current num_hands=1 is fine

- **Q:** Is wink-to-click acceptable, or do you need blink detection instead?
  - **Wink acceptable:** Current system is fine
  - **Blink required:** Different blendshape logic; currently doesn't distinguish blink from wink

### 4. **Performance Targets**

- **Q:** What's the target FPS? (25–30 current, 40–50 reasonable, 60–90 requires GPU?)
  - **Affects:** Which optimizations are needed; ROI detection (Phase 2) vs GPU

- **Q:** What's the acceptable gaze latency? (100–150ms current)
  - **<50ms:** Kalman filtering necessary (Phase 2.2)
  - **<20ms:** May need GPU + streaming inference

- **Q:** What's the acceptable gesture response latency? (200–300ms current, 150ms with voting)
  - **<100ms:** State machine (Phase 1) instead of voting
  - **<50ms:** Real-time LSTM required; GPU recommended

### 5. **Infrastructure & Deployment**

- **Q:** Do you have Python ML infrastructure (PyTorch, TensorFlow, scikit-learn)?
  - **YES:** LSTM (Phase 3) and advanced models are feasible
  - **NO:** Skip Phase 3; focus on signal processing (Phase 1–2)

- **Q:** Is this for personal use or deployment to others?
  - **Personal:** Can skip LSTM; hand-tuned Phase 1–2 is sufficient
  - **Deployment:** Need LSTM (95%+ accuracy) + extensive testing

- **Q:** What's the maintenance model? (Single developer, team, open-source?)
  - **Single dev:** Keep Phase 1–2 simple; avoid LSTM complexity
  - **Team:** LSTM and structured testing (Phase 3) justified

---

## C. Metrics to Log Immediately

### Frame-Level Metrics (Every Frame)

1. **FPS:** `1.0 / elapsed_seconds_per_frame`
   - **Target:** 40–50 FPS
   - **Log:** Rolling 1-second average
   - **Tool:** Existing fps_display variable (expand to file logging)

2. **Detection Confidence:** Average MediaPipe confidence for face, iris, hand
   - **Format:** `[timestamp, face_confidence, iris_confidence, hand_confidence]`
   - **Log:** Every 30 frames (1 Hz at 30 FPS)
   - **Tool:** New `compute_detection_confidence()` function

3. **Gaze Stability:** Standard deviation of gaze position over 5-frame window
   - **Format:** `[timestamp, gaze_std_dev]`
   - **Log:** Every 30 frames
   - **Tool:** New `compute_gaze_stability()` function

4. **Stage Timing:** Per-stage execution time (face detect, hand detect, gaze process, gesture, display)
   - **Format:** `[timestamp, face_detect_ms, hand_detect_ms, gaze_ms, gesture_ms, display_ms]`
   - **Log:** Every 30 frames
   - **Tool:** PerformanceMonitor class (Action Item #4)

### Session-Level Metrics (Every 30s or End of Session)

1. **Gesture Trigger Count:** Total count of each gesture type fired
   - **Format:** `[timestamp, scroll_count, zoom_count, drag_count, swipe_count, palm_count]`
   - **Log:** Every 30 seconds
   - **Tool:** New `GestureCounter` class

2. **Gesture Accuracy:** False-positive rate (spurious triggers), false-negative rate (missed triggers)
   - **Measurement:** Manual annotation (impractical) or contextual heuristics (e.g., detect when hand is off-screen but gesture fires)
   - **Log:** End of session via post-hoc analysis
   - **Tool:** Logging infrastructure + post-hoc analyzer

3. **Calibration Drift:** Gaze error at center screen (every 5 minutes)
   - **Measurement:** Assume user is looking at screen center; compute offset
   - **Format:** `[timestamp, drift_x_px, drift_y_px]`
   - **Log:** Every 5 minutes
   - **Tool:** New `compute_drift()` function (already implemented in calibrator.correct_drift)

4. **System Health:** Frame drops, model failures, exceptions
   - **Format:** `[timestamp, exception_type, file, line]`
   - **Log:** On occurrence
   - **Tool:** Logging.exception() throughout codebase

### Visualization Metrics (For User Feedback)

1. **Gaze Heatmap:** Where is the user looking? (Useful for UX research)
   - **Tool:** Accumulate gaze_ratio over session; visualize as 2D heatmap

2. **Gesture Frequency Chart:** Which gestures are used most?
   - **Tool:** Bar chart of gesture_count

3. **Performance Timeline:** FPS, latency, confidence over time
   - **Tool:** Line chart with time on X-axis

---

## D. Suggested Testing Harness

### Unit Tests (Existing, Expand)

Current: 59 tests (processor, tracker, calibration, gestures)

**Add:**
- Test normalize_hand_landmarks() (normalize invariance)
- Test compute_gaze_confidence() (confidence computation)
- Test GestureVotingBuffer (voting logic)
- Test ROI extraction (bounding box computation)
- Test Kalman filter (convergence, outlier rejection)

**Effort:** 8–10 hours

### Integration Tests (New)

1. **End-to-End Gesture Test**
   - Setup: Display buttons; user performs gesture
   - Assertion: Button click fires
   - Variations: Fast gesture, slow gesture, transient gesture

2. **Gaze Tracking Under Lighting**
   - Setup: Vary lighting (normal, dim, bright, shadows)
   - Assertion: Gaze latency <100ms, accuracy ±50px
   - Metrics: Log FPS, confidence, stability

3. **Head Movement Robustness**
   - Setup: User turns head ±30°, ±45°, ±60°
   - Assertion: Gaze tracks correctly (with/without 3D compensation)
   - Metrics: Gaze error vs head angle

4. **Calibration Under Adversity**
   - Setup: Calibrate while blinking, moving, under dim light
   - Assertion: Calibration completes; gaze error <±50px
   - Metrics: Inlier count, homography quality

5. **Performance Regression Test**
   - Setup: Record 1-minute video; run system end-to-end
   - Assertion: FPS maintains ≥30, no crashes
   - Metrics: FPS histogram, CPU usage, memory

**Effort:** 16–20 hours (implementation + data collection)

### Benchmark Suite (New)

**Test Scenarios:**
1. Stationary gaze (no head movement, no hand)
2. Smooth gaze tracking (slow eye movement)
3. Rapid gaze (saccades, fast eye movement)
4. Gesture spam (rapid pinch-release-pinch)
5. Gesture under occlusion (hand partially off-screen)
6. Low-light tracking (dim room, <100 lux)
7. Calibration stability (calibrate 5 times; compare results)

**Metrics per Scenario:**
- FPS, latency, accuracy, stability, confidence

**Effort:** 12–16 hours (test infrastructure + data collection)

---

## E. Dependencies & Versions

### Current (From requirements.txt)

```
mediapipe>=0.10.9
numpy>=1.24
opencv-contrib-python>=4.8
pyautogui>=0.9.54
```

### Phase 2 Additions (Optional)

```
scipy>=1.10.0  # For Kalman filter, statistics
scikit-learn>=1.3.0  # For decision tree (if using ML-based finger counting)
```

### Phase 3 Additions (Optional)

```
torch>=2.0.0  # For LSTM gesture model
tensorboard>=2.10.0  # For training visualization
```

### GPU Acceleration (Optional)

```
onnx>=1.14.0  # For model export/import
onnxruntime-gpu>=1.16.0  # GPU inference runtime
# OR
tensorrt>=8.6.0  # Alternative GPU inference (NVIDIA only)
```

---

## F. Known Limitations

### Current System (Before Optimizations)

1. **Single Hand:** Only first detected hand is tracked (num_hands=1)
   - **Workaround:** Re-detect if second hand enters frame
   - **Fix:** Change num_hands=2; track separately

2. **Iris vs Pupil:** Uses MediaPipe iris landmarks, not actual pupil segmentation
   - **Limitation:** Fails in low light, with glasses, bright backlighting
   - **Workaround:** Add ellipse-fitting fallback (Problem 3A)

3. **Linear Head Compensation:** Only yaw and pitch; no roll; linear approximation
   - **Limitation:** Fails for head angles >±30°
   - **Workaround:** Add 3D head pose (Action Item #9)

4. **Static Homography:** Calibration is one-time; only translational drift correction
   - **Limitation:** Calibration errors compound if head pose changes
   - **Workaround:** Add recalibration hotkey (Action Item #8)

5. **No Gesture Confidence:** Gestures fire on finger count alone; no confidence score
   - **Limitation:** False gestures during transient hand states
   - **Workaround:** Add temporal voting (Action Item #2)

6. **No Multi-Modal Fusion:** Gaze and gesture are independent; no cross-channel validation
   - **Limitation:** Can't reject impossible gaze/gesture combinations
   - **Example:** Gaze at bottom-left + swipe-right-desktop (are both real?)

### After Phase 1–2 Optimizations

1. **Frame Rate Ceiling:** ~40–50 FPS on CPU (without ROI optimization, face detection every frame)
   - **Workaround:** GPU acceleration (Phase 3)

2. **Gesture Latency:** ~150ms (temporal voting over 5 frames)
   - **Workaround:** Shorter vote window (3 frames, less stable) or state machine

3. **Head Angle Limit:** ±45°–60° (with 3D compensation, Action Item #9)
   - **Workaround:** Full body tracking (out of scope)

4. **Low-Light Failure:** Iris landmarks collapse in <50 lux
   - **Workaround:** Infrared or ellipse-fitting fallback

---

## G. Future Research Directions

### Short-Term (1–2 months)

1. **Gaze-Based Attention:** Log gaze heatmap; analyze which UI elements user looks at
2. **Gesture Customization:** Allow users to define custom hand gestures
3. **Calibration Auto-Trigger:** Detect when calibration drifts; prompt recalibration
4. **Multi-Monitor Support:** Map iris-to-screen for multiple displays

### Medium-Term (3–6 months)

1. **Eye Contact Detection:** Detect when user is looking at camera (for video calls)
2. **Fatigue Detection:** Detect eye closure patterns indicative of fatigue/drowsiness
3. **Gaze-Contingent Rendering:** Render high-quality only at gaze point (save GPU)
4. **Gesture Personalization:** LSTM trained per-user for higher accuracy

### Long-Term (6–12 months)

1. **Multimodal Control:** Combine gaze + gesture + voice for richer interaction
2. **Accessibility Toolkit:** Package as library for assistive technology developers
3. **Privacy-Preserving Gaze:** Gaze processed on-device; no cloud transmission
4. **AR/VR Integration:** Adapt to AR headsets (different camera model, wider FOV)

---

## H. Glossary

| Term | Definition |
|------|-----------|
| **Blendshape** | Facial expression weight (0–1) from MediaPipe; e.g., eyeBlinkLeft |
| **Cooldown** | Minimum time between repeated gesture triggers |
| **Drift** | Gradual gaze offset over time (causes cursor to move away from fixation) |
| **EMA** | Exponential Moving Average; simple low-pass filter |
| **Gaze Ratio** | Normalized iris position within eye socket (0–1 range) |
| **Head Compensation** | Subtracting head rotation effect from gaze estimate |
| **Homography** | 3×3 matrix that maps points from one 2D plane to another (iris-to-screen) |
| **Hysteresis** | Requiring sustained state change before triggering action |
| **Inlier** | Data point that fits the model (vs outlier) |
| **Kalman Filter** | Bayesian filter that estimates state from noisy measurements |
| **Landmark** | (x, y, z) position of a facial or hand feature |
| **MAD** | Median Absolute Deviation; robust outlier detection |
| **One-Euro Filter** | Adaptive low-pass filter; smoother at rest, less lag during motion |
| **RANSAC** | Robust algorithm to estimate model from data with outliers |
| **ROI** | Region of Interest; cropped area around detected object |
| **Saccade** | Rapid eye movement (vs smooth pursuit) |
| **Threshold** | Cutoff value for decision (e.g., confidence > 0.7) |

---

## I. References & Further Reading

### MediaPipe

- [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
- [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [MediaPipe Gesture Recognizer](https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer)

### Gaze Tracking & Eye Tracking

- **Survey (2017):** [A Review and Analysis of Eye-Gaze Estimation Systems](https://arxiv.org/pdf/1708.01817)
- **Recent (2024):** [Robust Camera-Based Eye-Tracking with Head Movements](https://pmc.ncbi.nlm.nih.gov/articles/PMC12734114/)
- **Kalman Filtering (2024):** [MediaPipe Iris and Kalman Filter for Robust Eye Gaze Tracking](https://www.atlantis-press.com/proceedings/icsice-24/126011300)

### Hand Gesture Recognition

- **LSTM+CNN (2024):** [Real-Time Hand Gesture Recognition Using LSTM](https://arxiv.org/html/2506.11154v1)
- **Transformer (2024):** [Dynamic Gesture Recognition Using Transformer and MediaPipe](https://www.mdpi.com/2673-4591/108/1/22)
- **Temporal Consistency (2024):** [MediaPipe Temporal Consistency with Kalman Filtering](https://dl.acm.org/doi/full/10.1145/3703187.3703295)

### Pupil Detection

- **Open Iris Framework (2024):** [Open Iris—An Open Source Framework for Eye-Tracking](https://pmc.ncbi.nlm.nih.gov/articles/PMC10925248/)
- **Ellipse Fitting:** [Pupil and Iris Detection Using OpenCV](https://www.researchgate.net/publication/261313914_Pupil_and_Iris_Detection_in_Dynamic_Pupillometry_Using_the_OpenCV_Library)

### Performance Optimization

- **MediaPipe ROI Tracking:** [MediaPipe Holistic—Simultaneous Face, Hand, and Pose](https://research.google/blog/mediapipe-holistic-simultaneous-face-hand-and-pose-prediction-on-device/)
- **One-Euro Filter:** [1€ Filter: A Simple Speed-Based Low-Pass Filter for Noisy Input](https://cristal.univ-lille.fr/~casiez/1euro/)

---

## J. Contact & Support

### For This Project

- **Repository:** [GitHub: ZeroTouch](https://github.com/Wissniper/Gesture-And-Eye-Track-System)
- **Issues:** Report bugs via GitHub issues
- **Discussions:** Use GitHub discussions for design questions

### For Dependencies

- **MediaPipe:** https://github.com/google/mediapipe
- **OpenCV:** https://github.com/opencv/opencv
- **PyAutoGUI:** https://github.com/asweigart/pyautogui

---

**End of Audit Report**

*Generated:* 2026-04-17  
*Auditor:* Claude Code (Repository Audit Framework)  
*Total Time Investment:* 6 research queries + codebase inspection  
*Output:* 8 detailed documents (01–08), ~15,000 lines total

