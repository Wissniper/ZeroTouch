# Requirements: IrisFlow C++ & Audit Fix

**Defined**: 2026-05-03
**Core Value**: High-performance, stable gaze/gesture control.

## v1 Requirements (Rewrite + Core Fixes)

### Core System (C++)
- [ ] **CORE-01**: CMake project structure with ONNX Runtime & OpenCV dependencies
- [ ] **CORE-02**: Multi-threaded capture pipeline (Producer/Consumer)
- [ ] **CORE-03**: OS-level cursor/keyboard event injection (macOS/Linux)

### Audit Remediation & Stability
- [ ] **STAB-01**: Confidence Gating: Ignore detections with confidence < 0.7
- [ ] **STAB-02**: Temporal Filtering: 3-5 frame majority voting for gestures
- [ ] **STAB-03**: Kalman Filter: Replace One-Euro with Kalman for gaze trajectory
- [ ] **STAB-04**: ROI Tracking: Only process relevant sub-regions for 2x FPS gain
- [ ] **STAB-05**: Normalized Finger Counting: Hand-scale invariant heuristics

### Gaze & Calibration
- [ ] **GAZE-01**: ONNX Inference for iris landmarks and gaze vector
- [ ] **GAZE-02**: 9-point Homography Calibration with RANSAC outlier rejection
- [ ] **GAZE-03**: Head-Pose Compensation (Decoupling head movement from gaze)

### Gesture Control
- [ ] **GEST-01**: Wink Detection (Differential blink score)
- [ ] **GEST-02**: Multi-finger gesture recognition (Scroll, Zoom, Drag)

## v2 Requirements (Expansion)
- **EXP-01**: LSTM/CNN Temporal Gesture Model (Moving beyond heuristics)
- **EXP-02**: Direct Pupil Segmentation (Fallback for iris landmarks)
- **EXP-03**: Full 3D Head Pose Model

## Out of Scope
- Windows Support (Phase 1)
- In-app Model Training

## Traceability
(To be populated by Roadmap)

## Traceability (Updated)

| Requirement | Phase | Status |
|-------------|-------|--------|
| CORE-01 | Phase 1 | Pending |
| CORE-02 | Phase 1 | Pending |
| CORE-03 | Phase 5 | Pending |
| STAB-01 | Phase 2 | Pending |
| STAB-02 | Phase 4 | Pending |
| STAB-03 | Phase 3 | Pending |
| STAB-04 | Phase 2 | Pending |
| STAB-05 | Phase 4 | Pending |
| GAZE-01 | Phase 2 | Pending |
| GAZE-02 | Phase 3 | Pending |
| GAZE-03 | Phase 3 | Pending |
| GEST-01 | Phase 4 | Pending |
| GEST-02 | Phase 4 | Pending |
