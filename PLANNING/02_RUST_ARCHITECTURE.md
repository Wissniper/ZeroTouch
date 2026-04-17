# IrisFlow Rust: System Architecture & Design

This document outlines the Rust implementation architecture, module structure, data flow, and key design decisions.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      IrisFlow Rust Runtime                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌───────────────────┐     ┌─────────────────┐
│   Webcam     │─────►│ MediaPipe Face +  │────►│  Gaze ONNX Model│
│ (v4l2/AVF)   │      │ Hand Detection    │     │  (landmarks→xy) │
└──────────────┘      │ (subprocess or    │     └────────┬─────────┘
                      │  Rust bindings)   │              │
                      └───────────────────┘              │
                                                         ▼
┌──────────────┐     ┌───────────────────┐     ┌─────────────────┐
│  PyAutoGUI   │◄────│  Gaze Smoothing   │◄────│  Landmark       │
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
src/
├── main.rs                      # CLI entry point, config loading, startup
├── lib.rs                       # Re-exports public modules
├── config.rs                    # Configuration struct, parsing
├── error.rs                     # Custom error types (Error enum)
├── metrics.rs                   # FPS, latency counters, logging
│
├── camera/                      # Frame capture pipeline
│   ├── mod.rs                   # Camera trait, platform selection
│   ├── macos.rs                 # AVFoundation implementation
│   ├── linux.rs                 # v4l2 implementation
│   └── buffer.rs                # Ring buffer (unsafe Rust)
│
├── detection/                   # MediaPipe landmark extraction
│   ├── mod.rs                   # Detection trait, result types
│   ├── mediapipe.rs             # MediaPipe wrapper (subprocess or FFI)
│   └── confidence.rs            # Confidence gating, quality checks
│
├── inference/                   # Model loading and inference
│   ├── mod.rs                   # Inference session management
│   ├── gaze.rs                  # Gaze regression model
│   ├── gesture.rs               # Gesture LSTM model
│   └── onnx_utils.rs            # ONNX Runtime helpers
│
├── processing/                  # Preprocessing and smoothing
│   ├── mod.rs                   # Main processor pipeline
│   ├── landmarks.rs             # Landmark normalization, feature extraction
│   ├── smoothing.rs             # One-Euro filter (Rust port)
│   └── kalman.rs                # Kalman filter (optional)
│
├── gestures/                    # Gesture recognition state machine
│   ├── mod.rs                   # Gesture controller, state machine
│   ├── detector.rs              # Frame-by-frame detection
│   └── buffer.rs                # 10-frame gesture buffer
│
├── control/                     # Desktop control (mouse, keyboard, window mgmt)
│   ├── mod.rs                   # Control trait, error handling
│   ├── macos.rs                 # AppleScript + native APIs
│   ├── linux.rs                 # wmctrl, xdotool, etc.
│   └── mouse.rs                 # Mouse movement and clicking
│
├── pipeline.rs                  # Main real-time event loop
└── calibration.rs               # Homography calibration, drift correction
```

---

## Key Data Structures

### Camera Frame
```rust
pub struct Frame {
    pub data: Vec<u8>,              // Raw RGB bytes
    pub width: u32,
    pub height: u32,
    pub timestamp: f64,             // Seconds since start
    pub frame_number: u64,
}
```

### Face/Hand Detection Results
```rust
pub struct FaceLandmarks {
    pub landmarks: Vec<[f32; 3]>,   // 468 points (x, y, z)
    pub confidence: f32,             // 0.0-1.0, gate on > 0.7
}

pub struct HandLandmarks {
    pub landmarks: Vec<[f32; 3]>,   // 21 points (x, y, z)
    pub handedness: Handedness,     // Left or Right
    pub confidence: f32,
}

pub enum Handedness {
    Left,
    Right,
}
```

### Gaze Inference Result
```rust
pub struct GazeResult {
    pub x: f32,                     // Screen X coordinate (0-1920)
    pub y: f32,                     // Screen Y coordinate (0-1080)
    pub confidence: f32,            // Model confidence (0.0-1.0)
    pub latency_ms: f32,            // Inference latency
}
```

### Gesture Recognition Result
```rust
pub enum Gesture {
    OpenHand,
    Pinch,
    ScrollReady,
    Swipe,
    Palm,
    Idle,
}

pub struct GestureResult {
    pub gesture: Gesture,
    pub confidence: f32,            // Model confidence
    pub triggered: bool,            // After temporal filtering + cooldown
    pub latency_ms: f32,
}
```

---

## Core Algorithms

### 1. One-Euro Filter (Rust Port)

**Purpose:** Smooth gaze coordinates with adaptive low-pass filtering.

```rust
pub struct OneEuroFilter {
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
    freq: f32,
    
    x_prev: f32,
    dx_prev: f32,
    t_prev: f32,
}

impl OneEuroFilter {
    pub fn new(min_cutoff: f32, beta: f32, d_cutoff: f32, freq: f32) -> Self {
        // ...
    }
    
    pub fn filter(&mut self, x: f32, t: f32) -> f32 {
        let dx = (x - self.x_prev) / (t - self.t_prev);
        let dx_filtered = self.alpha(self.d_cutoff) * dx + 
                         (1.0 - self.alpha(self.d_cutoff)) * self.dx_prev;
        
        let cutoff = self.min_cutoff + self.beta * dx_filtered.abs();
        let alpha = self.alpha(cutoff);
        let x_filtered = alpha * x + (1.0 - alpha) * self.x_prev;
        
        self.x_prev = x_filtered;
        self.dx_prev = dx_filtered;
        self.t_prev = t;
        
        x_filtered
    }
    
    fn alpha(&self, cutoff: f32) -> f32 {
        let tau = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
        let dt = 1.0 / self.freq;
        dt / (tau + dt)
    }
}
```

### 2. Temporal Gesture Filtering

**Purpose:** Gate gesture detection on stable, high-confidence classifications over multiple frames.

```rust
pub struct TemporalGestureFilter {
    buffer: VecDeque<Gesture>,      // Last N gesture predictions
    window_size: usize,             // Typically 5 frames
    cooldown_ms: u64,
    last_trigger_time: Instant,
}

impl TemporalGestureFilter {
    pub fn update(&mut self, gesture: Gesture) -> Option<Gesture> {
        self.buffer.push_back(gesture);
        if self.buffer.len() > self.window_size {
            self.buffer.pop_front();
        }
        
        // Majority voting
        if self.buffer.len() == self.window_size {
            let counts = /* count occurrences of each gesture */;
            let most_common = counts.iter().max_by_key(|&count| count);
            
            if self.should_trigger(most_common) {
                self.last_trigger_time = Instant::now();
                return Some(most_common);
            }
        }
        None
    }
    
    fn should_trigger(&self, gesture: Gesture) -> bool {
        self.last_trigger_time.elapsed().as_millis() as u64 >= self.cooldown_ms
    }
}
```

### 3. Gesture Buffer (10-Frame Window)

```rust
pub struct GestureBuffer {
    landmarks: VecDeque<Vec<f32>>,  // 10-frame window of hand landmarks
    max_size: usize,                 // = 10
}

impl GestureBuffer {
    pub fn push(&mut self, landmarks: Vec<f32>) {
        self.landmarks.push_back(landmarks);
        if self.landmarks.len() > self.max_size {
            self.landmarks.pop_front();
        }
    }
    
    pub fn ready(&self) -> bool {
        self.landmarks.len() == self.max_size
    }
    
    pub fn as_tensor(&self) -> Tensor {
        // Convert to [1, 630] tensor for LSTM inference
        let data: Vec<f32> = self.landmarks.iter().flatten().copied().collect();
        Tensor::of_slice(&data).reshape(&[1, 630])
    }
}
```

---

## Real-Time Pipeline: Execution Flow

### Main Event Loop (target: <16ms per frame)

```rust
pub struct IrisFlowPipeline {
    camera: Box<dyn Camera>,
    detector: MediaPipeDetector,
    gaze_model: GazeInference,
    gesture_model: GestureInference,
    gaze_processor: GazeProcessor,
    gesture_controller: GestureController,
    desktop_control: Box<dyn DesktopControl>,
    metrics: Metrics,
}

impl IrisFlowPipeline {
    pub async fn run(&mut self) -> Result<()> {
        loop {
            // Frame capture (4ms target)
            let frame = self.camera.capture()?;
            
            // Detection (5ms target)
            let t0 = Instant::now();
            let face = self.detector.detect_face(&frame)?;
            let hands = self.detector.detect_hands(&frame)?;
            let detection_ms = t0.elapsed().as_secs_f32() * 1000.0;
            
            // Gaze pipeline (2ms target)
            if let Some(face_lms) = face {
                let t1 = Instant::now();
                let gaze = self.gaze_model.infer(&face_lms)?;
                let gaze_ms = t1.elapsed().as_secs_f32() * 1000.0;
                
                // Smooth + output
                let smoothed = self.gaze_processor.smooth(gaze);
                self.desktop_control.move_cursor(smoothed.x, smoothed.y)?;
                
                // Wink detection (if confidence > 0.7)
                if face_lms.confidence > 0.7 {
                    if let Some(wink) = self.gesture_controller.detect_wink(&face_lms) {
                        self.desktop_control.click(wink)?;
                    }
                }
            }
            
            // Gesture pipeline (3ms target)
            if let Some(hand_lms) = hands.first() {
                self.gesture_controller.push_landmarks(hand_lms);
                
                if let Some(gesture) = self.gesture_controller.detect() {
                    match gesture {
                        Gesture::Pinch => self.desktop_control.zoom_in()?,
                        Gesture::ScrollReady => self.desktop_control.scroll()?,
                        Gesture::Swipe => self.desktop_control.switch_desktop()?,
                        _ => {}
                    }
                }
            }
            
            // Metrics
            let frame_time_ms = frame.timestamp * 1000.0;
            self.metrics.record_frame_time(frame_time_ms);
            
            if self.metrics.frame_count % 30 == 0 {
                eprintln!("FPS: {:.1}, Latency: {:.1}ms", 
                         self.metrics.fps(), 
                         self.metrics.avg_latency_ms());
            }
        }
    }
}
```

---

## Memory Management & Performance

### Frame Buffer Ring (Unsafe Rust)

The ring buffer uses unsafe Rust for zero-copy frame storage:

```rust
pub struct FrameRingBuffer {
    data: Vec<u8>,
    capacity: usize,
    write_pos: usize,
    read_pos: usize,
}

impl FrameRingBuffer {
    pub unsafe fn write(&mut self, frame: &[u8]) {
        // Copy frame into circular buffer without allocation
        let remaining = self.capacity - self.write_pos;
        if frame.len() <= remaining {
            std::ptr::copy_nonoverlapping(
                frame.as_ptr(),
                self.data.as_mut_ptr().add(self.write_pos),
                frame.len(),
            );
        } else {
            // Wrap around
            std::ptr::copy_nonoverlapping(
                frame.as_ptr(),
                self.data.as_mut_ptr().add(self.write_pos),
                remaining,
            );
            std::ptr::copy_nonoverlapping(
                frame.as_ptr().add(remaining),
                self.data.as_mut_ptr(),
                frame.len() - remaining,
            );
        }
        self.write_pos = (self.write_pos + frame.len()) % self.capacity;
    }
}
```

### Memory Pooling

Pre-allocate tensors before real-time loop to avoid GC pauses:

```rust
pub struct MemoryPool {
    gaze_input_tensors: Vec<Tensor>,    // Pre-allocated for inference
    gesture_input_tensors: Vec<Tensor>,
    available: Vec<usize>,              // Indices of available tensors
}

impl MemoryPool {
    pub fn acquire(&mut self) -> Option<&mut Tensor> {
        self.available.pop().map(|idx| &mut self.gaze_input_tensors[idx])
    }
    
    pub fn release(&mut self, idx: usize) {
        self.available.push(idx);
    }
}
```

---

## Error Handling

```rust
#[derive(Debug)]
pub enum IrisFlowError {
    // Camera errors
    CameraNotAvailable,
    CaptureError(String),
    
    // Detection errors
    DetectionFailed(String),
    MediaPipeError(String),
    
    // Inference errors
    ModelLoadError(String),
    InferenceError(String),
    
    // Desktop control errors
    PermissionDenied,
    MouseControlFailed,
    
    // General
    ConfigError(String),
    Io(std::io::Error),
}

impl Display for IrisFlowError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            IrisFlowError::CameraNotAvailable => write!(f, "Camera not available"),
            IrisFlowError::DetectionFailed(msg) => write!(f, "Detection failed: {}", msg),
            // ... other variants
        }
    }
}

pub type Result<T> = std::result::Result<T, IrisFlowError>;
```

---

## Concurrency Model

**Single-threaded + async** approach:

- Main thread: Real-time event loop (camera → inference → control)
- Tokio async: Background tasks (config reloading, metrics flushing)
- NO explicit threading to avoid context switching overhead in hot loop

```rust
#[tokio::main]
async fn main() -> Result<()> {
    let mut pipeline = IrisFlowPipeline::new(config)?;
    
    // Background metric flush task
    tokio::spawn(async {
        loop {
            tokio::time::sleep(Duration::from_secs(10)).await;
            metrics.flush_to_disk();
        }
    });
    
    // Main real-time loop (blocks)
    pipeline.run().await
}
```

---

## Testing Strategy

### Unit Tests
- Frame buffer wraparound
- One-Euro filter smoothing
- Gesture temporal filtering
- Coordinate transforms

### Integration Tests
- End-to-end pipeline with mock camera
- Inference latency measurement
- Memory stability (30 min sustained run)

### Benchmarks (Criterion)
- `bench_inference_gaze` — Per-frame gaze inference latency
- `bench_inference_gesture` — 10-frame gesture inference latency
- `bench_frame_capture` — Camera frame capture jitter
- `bench_smoothing` — One-Euro filter overhead

---

## Platform-Specific Implementation

### macOS
- Camera: AVFoundation (low latency, native)
- Control: Quartz Event Services (mouse), AppleScript (window switching)
- Accessibility: Request permission at startup

### Linux
- Camera: Video4Linux (v4l2)
- Control: xdotool (mouse, keyboard), wmctrl (window management)
- Dependencies: libv4l, xdotool package

---

## Performance Targets

| Component | Target | Measurement |
|-----------|--------|------------|
| Frame capture | <5ms | Camera timestamp to buffer |
| Detection | <8ms | MediaPipe face+hand inference |
| Gaze inference | <2ms | ONNX Runtime on GPU |
| Gesture inference | <2ms | ONNX Runtime on GPU |
| Gaze smoothing | <1ms | One-Euro filter |
| Desktop control | <1ms | Mouse move syscall |
| **Total per frame** | **<16ms** | 60 FPS sustained |

---

## Optimization Checklist

- [ ] Pre-allocate all buffers before real-time loop
- [ ] Zero-copy frame transfers (ring buffer)
- [ ] GPU inference (ONNX Runtime on CUDA/Metal)
- [ ] Batch processing if multiple frames queued
- [ ] SIMD for landmark preprocessing (normalize, distance calc)
- [ ] Profile with flamegraph; identify bottlenecks
- [ ] Reduce allocations in hot path (use arena allocators if needed)
- [ ] No logging in real-time loop (deferred to background thread)

---

## Deployment Checklist

- [ ] Cross-platform build (macOS x86_64 + ARM64, Linux)
- [ ] ONNX models bundled in release binary or external
- [ ] Configuration file defaults (no hard-coded paths)
- [ ] Graceful shutdown (SIGTERM handler)
- [ ] Logging to file (optional, off by default)
