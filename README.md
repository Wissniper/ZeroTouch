# IrisFlow: Real-Time Gaze & Gesture Interface (C++)

IrisFlow is a high-performance, zero-touch computer interface that replaces the mouse with eye gaze and hand gestures. This version is a complete C++17 rewrite of the original Python prototype, optimized for ultra-low latency and production stability.

## Key Features

- **60+ FPS Tracking**: Multi-threaded capture and inference pipeline.
- **Kalman Filtering**: Professional-grade gaze stabilization to eliminate jitter.
- **Confidence Gating**: All detections are validated against confidence thresholds to prevent false triggers.
- **ROI Tracking**: Optimizes performance by only processing relevant frame regions.
- **Multi-OS Support**: macOS (Quartz) and Linux (X11/Xtest) event injection.
- **Robust Gestures**: Temporal majority voting for stable wink and hand gesture detection.

## Project Structure

- `irisflow-cpp/`: Main C++ project root.
  - `include/`: Header files organized by module (camera, inference, processing, control).
  - `src/`: Implementation files.
  - `models/`: ONNX models for gaze and gesture inference.

## Getting Started

### Dependencies

- **C++17 Compiler**
- **CMake 3.14+**
- **OpenCV**
- **ONNX Runtime**

### Build

```bash
cd irisflow-cpp
mkdir build && cd build
cmake ..
make
```

### Run

```bash
./irisflow
```

## Audit & Stability

This implementation addresses all critical failure modes identified in the project audit:

- Hand Gesture Instability (Fixed via Temporal Filtering)
- Finger Counting Unreliability (Fixed via Normalized Heuristics)
- Detection Jitter (Fixed via Kalman Filtering)
- Performance Bottlenecks (Fixed via C++ and ROI Tracking)

## License

[MIT](LICENSE) — Wisdom Ononiba, 2026