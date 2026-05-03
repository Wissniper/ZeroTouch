# GSD Project Guide: IrisFlow C++ Rewrite

## Context
This project is a high-performance C++17 rewrite of IrisFlow, focusing on real-time gaze tracking and hand gesture control. It explicitly addresses stability and performance issues identified in the `AUDIT_REPORT`.

## Workflow Enforcement
- **Phase Management**: All work MUST follow the phased roadmap in `.planning/ROADMAP.md`.
- **Project State**: Update `.planning/STATE.md` after every significant task.
- **Decision Log**: Log architectural decisions in `.planning/PROJECT.md`.
- **Command Entry**: Use `/gsd-plan-phase <N>` to start a phase.

## Tech Stack
- **Language**: C++17
- **Inference**: ONNX Runtime
- **Vision**: OpenCV
- **OS Control**: macOS Quartz / Linux X11

## Audit Focus Areas (MANDATORY)
- **Temporal Stability**: Majority voting for gestures (STAB-02).
- **Gaze Smoothing**: Kalman Filter (STAB-03).
- **Confidence**: All detections must be gated (STAB-01).
- **Performance**: ROI tracking (STAB-04).

---
*See .planning/ for full documentation.*
