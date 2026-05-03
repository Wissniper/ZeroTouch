# IrisFlow: C++ + ML Rewrite Master Plan

**Project Goal:** Rewrite IrisFlow from Python to C++17 with self-trained ML models, following the industry "Golden Path":
1. Train in Python (using PyTorch/TensorFlow).
2. Export to ONNX or TorchScript.
3. Run Production Inference in C++ (using ONNX Runtime or LibTorch).

Combining systems programming and machine learning learning objectives.

**Timeline:** 12-16 weeks (~300-400 hours)  
**Start Date:** 2026-04-17  
**Primary Hardware:** Desktop PC (Ryzen 9 5900X, RTX 3070 Ti) for training; M4 Mac for development/testing  
**Final Deliverable:** Standalone C++ binary, production-ready, 60+ FPS, <50px gaze error

---

## Project Phases

### Phase 1: Python ML Foundation (Weeks 1-4, ~120 hours)

[... Phase 1 remains the same as in the original plan ...]

---

### Phase 2: C++ Real-Time System (Weeks 5-12, ~180 hours)

**Goal:** Build a production C++ binary that captures frames, runs inference, and controls the desktop. Optimize for 60 FPS and <16ms latency per frame.

#### 2.1 Project Setup + Tooling (Week 5, ~20 hours)

**Create C++ project structure:**
```
irisflow-cpp/
├── CMakeLists.txt
├── include/
│   └── irisflow/
│       ├── camera/
│       ├── inference/
│       ├── processing/
│       ├── control/
│       └── core/
├── src/
│   ├── main.cpp
│   ├── camera/
│   ├── inference/
│   ├── processing/
│   ├── control/
│   └── core/
├── tests/
├── benchmarks/
└── models/
```

**Dependencies:**
- `onnxruntime` (C++ API) — inference
- `OpenCV` — webcam capture and image processing
- `nlohmann/json` — config parsing
- `CLI11` — CLI argument parsing
- `spdlog` — logging
- `GTest` — unit testing
- `google-benchmark` — performance benchmarking

**Deliverables:**
- `CMakeLists.txt` with all dependencies
- CI setup (GitHub Actions for C++ build on macOS/Linux)
- Development environment doc (vcpkg/conan setup)

#### 2.2 Webcam + Frame Capture Pipeline (Days 22-28, ~25 hours)

**Goal:** Capture frames at 30 FPS with minimal latency, zero-copy if possible.

- **Frame Capture Module** (`src/camera/`):
  - Interface: `ICamera` base class
  - macOS: AVFoundation (Objective-C++ bridge)
  - Linux: Video4Linux2 (raw ioctls)
  - Memory Management: Ring buffer with `std::unique_ptr`
  
- **Memory Management:**
  - Pre-allocate frame buffers before real-time loop
  - Use `cv::Mat` headers to avoid deep copies of image data

**Deliverables:**
- `src/camera/` implementations
- `tests/camera_tests.cpp`

#### 2.3 Model Loading + Inference (Days 29-35, ~25 hours)

**Goal:** Load ONNX models and run inference with <5ms latency using C++ API.

- **Model Loading:**
  - Load ONNX at startup (`gaze.onnx`, `gesture.onnx`)
  - Use `Ort::Env` and `Ort::Session`
  - Handle exceptions and error codes

- **Performance:**
  - Benchmark: time per inference (target: <2ms GPU, <5ms CPU)
  - Profile: Tensor creation overhead

**Deliverables:**
- `src/inference/` wrappers for ONNX Runtime
- `benchmarks/inference_bench.cpp`

#### 2.4 Frame Preprocessing + Landmark Extraction (Days 36-42, ~20 hours)

**Goal:** Extract MediaPipe landmarks from raw frames using C++ SDK.

- **MediaPipe Integration:**
  - Use MediaPipe C++ SDK if possible, or subprocess fallback
  - Extract 468 face landmarks and 21 hand landmarks

- **Landmark Smoothing:**
  - C++ implementation of One-Euro filter
  - SIMD optimizations for normalization if needed

#### 2.5 Real-Time Pipeline + Event Loop (Days 43-49, ~20 hours)

**Goal:** Orchestrate capture → detection → inference → control in <16ms per frame.

- **Main Loop** (`src/core/Pipeline.cpp`):
  - Synchronous hot loop for latency
  - Background thread for non-critical tasks (metrics, logging)
  - State machine for gesture recognition

#### 2.6 Desktop Control (Days 50-56, ~15 hours)

- **Control Module:**
  - macOS: Quartz Event Services (Core Graphics)
  - Linux: X11/Xtest or Wayland protocols
  - Move cursor, trigger clicks, handle gestures

[... Rest of document remains consistent with C++ focus ...]
