# IrisFlow (C++)

A C++17 framework for building a zero-touch computer interface driven by eye gaze and hand gestures. The pipeline skeleton is complete and builds cleanly; a trained inference model is not yet wired in.

## What's implemented

- **Threaded webcam capture** — background capture thread with mutex-gated frame handoff
- **Kalman filter gaze stabilisation** — 4-state (x, y, vx, vy) filter to reduce jitter
- **Homography calibration** — maps normalised gaze coordinates to screen coordinates via RANSAC homography
- **Gesture majority voting** — 5-frame temporal filter to debounce gesture events
- **OS event injection** — macOS (Quartz CGEvent) cursor move; gesture stubs ready to fill in
- **Head-pose compensation** — linear yaw/pitch correction applied before mapping

## What's not implemented yet

- A real gaze / hand-landmark inference model — the main loop currently uses the frame centre as a synthetic gaze point
- Wink and hand-gesture detection (both require live landmark output from a model)

## Build

Dependencies: C++17 compiler, CMake 3.14+, OpenCV (e.g. `brew install opencv`).

```bash
cd irisflow-cpp
cmake -B build .
cmake --build build
./build/irisflow
```

## Project structure

```
irisflow-cpp/
  include/irisflow/
    camera/       WebcamCapture
    processing/   KalmanFilter2D, GazeCalibrator
    control/      GestureController, OSEventInjector
    core/         Types (DetectionResult, Landmark)
  src/            Implementations
```

## License

[MIT](LICENSE) — Wisdom Ononiba, 2026
