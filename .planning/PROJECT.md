# IrisFlow C++ Rewrite & Audit Remediation

## What This Is
A high-performance rewrite of the IrisFlow zero-touch interface from Python to C++17, simultaneously addressing the critical stability and accuracy issues identified in the Project Audit. The system provides iris gaze tracking and hand gesture recognition at 60+ FPS with sub-16ms latency.

## Core Value
High-performance, stable, and production-ready gaze/gesture control that eliminates the jitter and false triggers of the prototype.

## Requirements

### Validated
(None yet)

### Active
- [ ] **Audit Fix**: Implement Temporal Gesture Filtering (Majority voting/Debouncing)
- [ ] **Audit Fix**: Confidence Gating for all MediaPipe detections
- [ ] **Audit Fix**: Improved Finger Counting (Normalization/Scale Invariance)
- [ ] **Audit Fix**: ROI Tracking for performance optimization
- [ ] **Audit Fix**: Kalman Filtering for Gaze stabilization
- [ ] Production C++ project structure (CMake)
- [ ] High-frequency webcam capture pipeline (macOS/Linux)
- [ ] ONNX Runtime inference engine integration
- [ ] OS-level event injection (macOS Quartz / Linux X11)
- [ ] Robust 9-point Homography Calibration with RANSAC

### Out of Scope
- [ ] Windows support (Phase 1 focus: macOS/Linux)
- [ ] Model training in C++

## Context
- The AUDIT_REPORT identified "Hand Gesture Instability" and "No Confidence Gating" as high-severity issues.
- The CPP_ML_REWRITE_PLAN.md provides the architectural "Golden Path".

## Constraints
- **Performance**: 60+ FPS, <16ms latency.
- **Reliability**: >95% gesture trigger accuracy.
- **Accuracy**: <20px gaze error (improved from prototype).

---
*Last updated: 2026-05-03 after audit integration*
