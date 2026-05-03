# IrisFlow C++: System Architecture & Design

This document outlines the C++17 implementation architecture, module structure, data flow, and key design decisions.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      IrisFlow C++ Runtime                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌───────────────────┐     ┌─────────────────┐
│   Webcam     │─────►│ MediaPipe Face +  │────►│  Gaze ONNX Model│
│ (v4l2/AVF)   │      │ Hand Detection    │     │  (landmarks→xy) │
└──────────────┘      │ (subprocess or    │     └────────┬─────────┘
                      │  C++ SDK)         │              │
                      └───────────────────┘              │
                                                         ▼
┌──────────────┐     ┌───────────────────┐     ┌─────────────────┐
│   UI Control │◄────│  Gaze Smoothing   │◄────│  Landmark       │
│  move/click  │     │  (One-Euro Filter)│     │  Preprocessing  │
└──────────────┘     └───────────────────┘     └─────────────────┘

┌──────────────────────────────────────────┐
│  Gesture Detection + State Machine       │
│  - 10-frame buffer of hand landmarks     │
│  - LSTM inference on buffer              │
│  - Temporal filtering (majority vote)    │
│  - Cooldown between gestures             │
└──────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│  Desktop Control                         │
│  - Mouse: move, click, scroll            │
│  - Keyboard: hotkeys                     │
│  - Window management: switch desktop     │
└──────────────────────────────────────────┘
```

---

## Module Structure

```
.
├── CMakeLists.txt               # Build configuration
├── include/
│   └── irisflow/
│       ├── camera/              # Frame capture interfaces
│       ├── detection/           # MediaPipe wrappers
│       ├── inference/           # ONNX Runtime wrappers
│       ├── processing/          # Preprocessing & Filters
│       ├── control/             # Desktop OS abstraction
│       ├── core/                # Pipeline & Metrics
│       └── utils/               # Common helpers
└── src/
    ├── main.cpp                 # CLI entry point
    ├── camera/                  
    │   ├── ICamera.hpp          # Camera interface
    │   ├── MacosCamera.cpp      # AVFoundation
    │   └── LinuxCamera.cpp      # v4l2
    ├── detection/               
    ├── inference/               
    │   ├── GazeInference.cpp    # ONNX Gaze Model
    │   └── GestureInference.cpp # ONNX Gesture LSTM
    ├── processing/              
    │   ├── OneEuroFilter.cpp    
    │   └── LandmarkProcessor.cpp
    ├── control/                 
    └── core/                    
        └── Pipeline.cpp         # Main real-time event loop
```

---

## Key Data Structures

### Camera Frame
```cpp
struct Frame {
    std::vector<uint8_t> data;      // Raw RGB bytes
    uint32_t width;
    uint32_t height;
    double timestamp;               // Seconds since start
    uint64_t frame_number;
};
```

### Face/Hand Detection Results
```cpp
struct Landmark {
    float x, y, z;
};

struct FaceLandmarks {
    std::vector<Landmark> landmarks; // 468 points
    float confidence;                // 0.0-1.0
};

enum class Handedness {
    Left,
    Right
};

struct HandLandmarks {
    std::vector<Landmark> landmarks; // 21 points
    Handedness handedness;
    float confidence;
};
```

---

## Core Algorithms

### 1. One-Euro Filter (C++17)

```cpp
class OneEuroFilter {
public:
    OneEuroFilter(float min_cutoff, float beta, float d_cutoff, float freq)
        : min_cutoff_(min_cutoff), beta_(beta), d_cutoff_(d_cutoff), freq_(freq) {}

    float filter(float x, float t) {
        float dt = t - t_prev_;
        if (t_prev_ < 0) dt = 1.0f / freq_;

        float dx = (x - x_prev_) / dt;
        float edx = low_pass_filter(dx, dx_prev_, alpha(d_cutoff_));
        
        float cutoff = min_cutoff_ + beta_ * std::abs(edx);
        float ex = low_pass_filter(x, x_prev_, alpha(cutoff));

        x_prev_ = ex;
        dx_prev_ = edx;
        t_prev_ = t;
        return ex;
    }

private:
    float alpha(float cutoff) {
        float tau = 1.0f / (2.0f * M_PI * cutoff);
        float dt = 1.0f / freq_;
        return dt / (tau + dt);
    }

    float low_pass_filter(float x, float prev_x, float alpha) {
        return alpha * x + (1.0f - alpha) * prev_x;
    }

    float min_cutoff_, beta_, d_cutoff_, freq_;
    float x_prev_ = 0, dx_prev_ = 0, t_prev_ = -1;
};
```

### 2. Gesture Buffer (10-Frame Window)

```cpp
class GestureBuffer {
public:
    void push(const std::vector<float>& landmarks) {
        buffer_.push_back(landmarks);
        if (buffer_.size() > max_size_) {
            buffer_.pop_front();
        }
    }

    bool is_ready() const { return buffer_.size() == max_size_; }

    std::vector<float> flatten() const {
        std::vector<float> flat;
        for (const auto& frame : buffer_) {
            flat.insert(flat.end(), frame.begin(), frame.end());
        }
        return flat;
    }

private:
    std::deque<std::vector<float>> buffer_;
    size_t max_size_ = 10;
};
```

---

## Real-Time Pipeline: Execution Flow

### Main Event Loop

```cpp
class IrisFlowPipeline {
public:
    void run() {
        while (running_) {
            // 1. Capture
            auto frame = camera_->capture();
            
            // 2. Detect (MediaPipe)
            auto face = detector_->detect_face(frame);
            auto hands = detector_->detect_hands(frame);
            
            // 3. Gaze Inference (ONNX)
            if (face) {
                auto gaze = gaze_model_->infer(*face);
                auto smoothed = gaze_filter_.filter(gaze);
                controller_->move_cursor(smoothed.x, smoothed.y);
            }
            
            // 4. Gesture Inference
            if (!hands.empty()) {
                gesture_buffer_.push(preprocess(hands[0]));
                if (gesture_buffer_.is_ready()) {
                    auto gesture = gesture_model_->infer(gesture_buffer_.flatten());
                    handle_gesture(gesture);
                }
            }
        }
    }

private:
    std::unique_ptr<ICamera> camera_;
    std::unique_ptr<IDetector> detector_;
    std::unique_ptr<IInference> gaze_model_;
    // ... filters and controllers
};
```

---

## Memory Management & Performance

### RAII and Smart Pointers
Unlike Rust's borrow checker, C++ relies on **RAII (Resource Acquisition Is Initialization)**.
- `std::unique_ptr`: Use for exclusive ownership of models and camera resources.
- `std::shared_ptr`: Use only when resources must be shared across threads (e.g., shared metrics).
- `std::vector` / `std::array`: Standard containers for landmark data, ensuring automatic cleanup.

### Zero-Copy Optimization
Use `cv::Mat` (OpenCV) or raw pointers with custom deleters to pass image data from the camera to inference engines without redundant copies.

---

## Concurrency Model

Modern C++ threads (`std::thread`) and atomics (`std::atomic`) are used for concurrency.

```cpp
std::atomic<bool> running{true};

void background_metrics(Metrics& m) {
    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        m.flush();
    }
}

int main() {
    IrisFlowPipeline pipeline;
    std::thread metrics_thread(background_metrics, std::ref(pipeline.metrics()));
    
    pipeline.run(); // Main loop
    
    running = false;
    metrics_thread.join();
    return 0;
}
```

---

## Performance Targets

| Component | Target | Measurement |
|-----------|--------|------------|
| Frame capture | <5ms | OpenCV/AVF latency |
| Detection | <8ms | MediaPipe C++ SDK |
| Gaze inference | <2ms | ONNX Runtime (TensorRT/CoreML) |
| Gesture inference | <2ms | ONNX Runtime |
| **Total per frame** | **<16ms** | 60 FPS sustained |
