# Roadmap: IrisFlow C++ & Audit Remediation

## Overview
- **Status**: Planning
- **Goal**: 60 FPS production C++ eye-tracker with audit-verified stability.
- **Phases**: 5

## Phase 1: Scaffolding & Capture Pipeline
**Goal**: Establish the C++ foundation and high-speed frame ingestion.
- **Requirements**: CORE-01, CORE-02
- **Success Criteria**:
  - CMake builds and links ONNX Runtime/OpenCV.
  - Webcam capture reaches 60 FPS in a multi-threaded ring buffer.

## Phase 2: Inference Engine & Performance Fixes
**Goal**: Run ML models with audit-recommended performance optimizations.
- **Requirements**: GAZE-01, STAB-01, STAB-04
- **Success Criteria**:
  - ONNX models load and infer in <5ms.
  - Confidence gating filters low-quality frames (Audit Fix).
  - ROI tracking implemented for landmark regions (Audit Fix).

## Phase 3: Gaze Stabilization & Calibration
**Goal**: Reliable eye-to-screen mapping with head compensation.
- **Requirements**: GAZE-02, GAZE-03, STAB-03
- **Success Criteria**:
  - Kalman filter reduces gaze jitter by >30% (Audit Fix).
  - Homography maps gaze to 4K screen with <30px error.
  - Head pose does not drift cursor by more than 50px during rotation.

## Phase 4: Robust Gesture Controller
**Goal**: Stabilize hand gestures using temporal filtering and normalized heuristics.
- **Requirements**: GEST-01, GEST-02, STAB-02, STAB-05
- **Success Criteria**:
  - Temporal filtering eliminates frame-by-frame gesture flickering (Audit Fix).
  - Finger counting is invariant to hand distance (Audit Fix).
  - Wink and Pinch gestures work with >90% reliability.

## Phase 5: Integration & OS Control
**Goal**: Full "ZeroTouch" experience with system-level cursor/keyboard control.
- **Requirements**: CORE-03
- **Success Criteria**:
  - Cursor movement is fluid on macOS/Linux.
  - Gestures successfully trigger system scroll, zoom, and clicks.
  - System survives 1-hour "soak test" without memory leaks or FPS drops.

---
## Traceability Matrix (REQUIREMENTS.md Update)
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

