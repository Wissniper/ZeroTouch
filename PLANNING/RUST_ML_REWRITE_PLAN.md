# IrisFlow: Rust + ML Rewrite Master Plan

**Project Goal:** Rewrite IrisFlow from Python to Rust with self-trained ML models, combining systems programming and machine learning learning objectives.

**Timeline:** 12-16 weeks (~300-400 hours)  
**Start Date:** 2026-04-17  
**Primary Hardware:** Desktop PC (Ryzen 9 5900X, RTX 3070 Ti) for training; M4 Mac for development/testing  
**Final Deliverable:** Standalone Rust binary, production-ready, 60+ FPS, <50px gaze error

---

## Project Phases

### Phase 1: Python ML Foundation (Weeks 1-4, ~120 hours)

**Goal:** Train production-ready models in Python. Once exported to ONNX, Python is discarded.

#### 1.1 Data Collection (Week 1, ~20 hours)
- **What:** Collect your own ground-truth dataset for gaze and gesture recognition
- **Gaze Data:** Record 30-60 minutes of eye movement with known screen positions
  - Method: 9-point grid (like calibration), but record 5+ samples per point
  - Output: `(face_landmarks, screen_x, screen_y)` tuples
  - Target: 5000 samples across full screen, varied head poses
  
- **Gesture Data:** Record 30-60 minutes of hand gestures
  - Method: Controlled recording of each gesture type (pinch, scroll, swipe, palm, open)
  - Output: `(hand_landmarks_sequence[10], gesture_label)` tuples
  - Target: 2000+ labeled 10-frame windows (~20 per label)
  - Labels: {open_hand, pinch, scroll_ready, swipe, palm, idle}

- **Deliverables:**
  - `data/gaze_dataset.csv` — (face_landmarks_json, screen_x, screen_y)
  - `data/gesture_sequences.pkl` — List of (landmarks_10frame, label)
  - `data/README.md` — Collection methodology, lighting conditions, head pose ranges

#### 1.2 Exploratory Data Analysis (EDA) + Baselines (Days 4-7, ~15 hours)
- Load datasets, visualize distributions
- Test simple baselines:
  - Gaze: Linear regression on raw iris ratio (establish RMSE baseline)
  - Gesture: Simple MLP on averaged landmarks (establish accuracy baseline)
- Document findings: data quality, imbalance, outliers

#### 1.3 Gaze Regression Model (Week 2, ~30 hours)
- **Architecture:** Small fully-connected network OR shallow CNN on face landmarks
  - Input: 468 MediaPipe face landmarks (flattened to 468×3 = 1404 dims)
  - Output: (screen_x, screen_y) in [0, 1920]×[0, 1080] range
  - Architecture option A (simple): 1404 → 256 → 128 → 2 (ReLU, no dropout)
  - Architecture option B (better): Lightweight CNN on landmark heatmaps
  
- **Training:**
  - Split: 70% train, 15% val, 15% test
  - Loss: MSE (mean squared error in pixels)
  - Optimizer: Adam, LR=0.001
  - Early stopping on val loss
  - Target: <50 px RMSE on test set
  
- **Deliverables:**
  - `models/gaze_model.py` — Model definition
  - `notebooks/gaze_training.ipynb` — Full training loop with validation plots
  - `models/gaze_best.pt` — Trained PyTorch checkpoint
  - `reports/gaze_model_report.md` — Architecture, metrics, failure cases

#### 1.4 Temporal Gesture LSTM (Week 3, ~40 hours)
- **Architecture:** Sequence model on hand landmarks
  - Input: Sequence of 10 frames × 21 hand landmarks × 3 (x,y,z) = 630-dim sequences
  - LSTM: 630 → 128 units → output layer
  - Output: Softmax over 6 gesture classes
  
- **Training:**
  - Data augmentation: Time warping, hand scale/rotation jitter
  - Validation: Held-out sequences (not individual frames)
  - Loss: Cross-entropy
  - Target: >95% accuracy on test gestures
  
- **Deliverables:**
  - `models/gesture_lstm.py` — LSTM architecture
  - `notebooks/gesture_training.ipynb` — Data augmentation, training, confusion matrix
  - `models/gesture_best.pt` — Trained checkpoint
  - `reports/gesture_model_report.md` — Per-class accuracy, false positives/negatives

#### 1.5 ONNX Export + Validation (Days 18-21, ~15 hours)
- Convert both models to ONNX format (for Rust inference)
- Test inference on ONNX Runtime (CPU + GPU)
- Validate outputs match PyTorch exactly
- Document input/output shapes and normalization

**Deliverables:**
  - `models/gaze.onnx` — Gaze regression model
  - `models/gesture.onnx` — Gesture LSTM model
  - `models/model_manifest.json` — Metadata: input shapes, output ranges, preprocessing steps

---

### Phase 2: Rust Real-Time System (Weeks 5-12, ~180 hours)

**Goal:** Build a production Rust binary that captures frames, runs inference, and controls the desktop. Optimize for 60 FPS and <16ms latency per frame.

#### 2.1 Project Setup + Tooling (Week 5, ~20 hours)

**Create Rust project structure:**
```
irisflow-rs/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── camera/         # Frame capture
│   │   ├── mod.rs
│   │   ├── v4l2.rs     # Linux/Mac webcam
│   │   └── buffer.rs   # Ring buffer, memory pooling
│   ├── inference/      # Model loading and inference
│   │   ├── mod.rs
│   │   ├── gaze.rs
│   │   └── gesture.rs
│   ├── processing/     # Frame preprocessing, filtering
│   │   ├── mod.rs
│   │   ├── landmarks.rs
│   │   └── smoothing.rs
│   ├── control/        # Mouse/keyboard output
│   │   ├── mod.rs
│   │   └── desktop.rs
│   ├── pipeline.rs     # Main real-time loop
│   └── metrics.rs      # FPS, latency monitoring
├── benches/            # Performance benchmarks
└── models/             # ONNX files + metadata
```

**Dependencies:**
- `tch-rs` OR `ort` (ONNX Runtime) — inference
- `v4l2-rs` or `nokhwa` — webcam capture
- `image` — image processing
- `serde_json` — config parsing
- `tokio` — async runtime (optional, for threading)
- `criterion` — benchmarking

**Deliverables:**
- `Cargo.toml` with all dependencies
- CI setup (GitHub Actions for tests on M4 + Linux)
- Development environment doc (setup on M4 and PC)

#### 2.2 Webcam + Frame Capture Pipeline (Days 22-28, ~25 hours)

**Goal:** Capture frames at 30 FPS with minimal latency, zero-copy if possible.

- **Frame Capture Module** (`src/camera/`):
  - Platform detection: macOS AVFoundation vs Linux v4l2
  - Ring buffer (pre-allocated, 5-frame buffer)
  - Frame timestamp and metadata
  - Error handling: camera disconnection, permission denial
  
- **Memory Management:**
  - Pre-allocate buffers before real-time loop
  - Zero-copy between capture and preprocessing
  - Profile memory usage (target: <50 MB resident)

- **Testing:**
  - Unit tests: Buffer wraparound, frame ordering
  - Integration test: Capture 300 frames, verify timestamps monotonic

**Deliverables:**
- `src/camera/mod.rs` — Camera trait, platform implementations
- `src/camera/buffer.rs` — Ring buffer with unsafe Rust
- `tests/camera_tests.rs` — Frame capture tests
- `docs/camera_design.md` — Architecture and memory layout

#### 2.3 Model Loading + Inference (Days 29-35, ~25 hours)

**Goal:** Load ONNX models and run inference with <5ms latency.

- **Model Loading:**
  - Load ONNX at startup (gaze.onnx, gesture.onnx)
  - Verify input shapes match expected landmarks
  - Handle inference errors gracefully

- **Gaze Inference:**
  - Input: 468 landmarks → Output: (x, y) screen coords
  - Batch size 1 (single frame)
  - Normalize landmarks to [-1, 1] range
  
- **Gesture Inference:**
  - Input: 10-frame window of 21 hand landmarks
  - Output: 6-class probability distribution
  - Argmax + confidence threshold for gesture detection

- **Performance:**
  - Benchmark: time per inference (target: <2ms GPU, <5ms CPU)
  - Profile: Memory allocation, tensor creation

**Deliverables:**
- `src/inference/mod.rs` — Model loading, session management
- `src/inference/gaze.rs` — Gaze model wrapper
- `src/inference/gesture.rs` — Gesture model wrapper
- `benches/inference_bench.rs` — Latency measurements
- `docs/inference_notes.md` — Input normalization, output interpretation

#### 2.4 Frame Preprocessing + Landmark Extraction (Days 36-42, ~20 hours)

**Goal:** Extract MediaPipe landmarks from raw frames.

- **MediaPipe Integration:**
  - Use MediaPipe Rust bindings (if available) OR wrap Python via subprocess (not ideal, but fallback)
  - Better option: Use `mp-solution-rs` crate or port detection logic
  - Extract face landmarks (468) + hand landmarks (21 × 2 hands)
  
- **Preprocessing:**
  - Resize frame to 640×480
  - Convert BGR to RGB
  - Normalize to float [0, 1]
  - Detect confidence, gate on > 0.7

- **Landmark Smoothing:**
  - Port One-Euro filter from Python
  - Kalman filter option (if time permits)
  - Per-landmark smoothing

**Deliverables:**
- `src/processing/landmarks.rs` — Landmark extraction
- `src/processing/smoothing.rs` — One-Euro filter (Rust implementation)
- `tests/smoothing_tests.rs` — Filter validation
- `docs/preprocessing_notes.md` — MediaPipe integration strategy

#### 2.5 Real-Time Pipeline + Event Loop (Days 43-49, ~20 hours)

**Goal:** Orchestrate capture → detection → inference → control in <16ms per frame.

**Architecture:**
```
Frame capture (4ms)
    ↓
MediaPipe detection (5ms)
    ↓
Landmark smoothing (1ms)
    ↓
Gaze inference (2ms)
    ↓
Gesture inference (2ms)
    ↓
Desktop control (1ms)
────────────────────
Total target: <16ms (60 FPS)
```

- **Main Loop** (`src/pipeline.rs`):
  - Tokio async OR raw threading
  - Coordinated timing: frame capture triggers detection
  - Error recovery: Frame drops, inference failures
  - FPS/latency metrics per stage

- **Gesture State Machine:**
  - Temporal filtering (majority vote over 3-5 frames)
  - Cooldown between gestures (500ms)
  - Confidence gating on LSTM output

- **Gaze Control:**
  - Map inference output to screen coords
  - Clamp to display bounds
  - Smooth cursor movement with existing One-Euro filter

**Deliverables:**
- `src/pipeline.rs` — Main event loop
- `src/main.rs` — CLI entry point, config loading
- `src/metrics.rs` — FPS/latency instrumentation
- `tests/pipeline_tests.rs` — Integration tests
- Latency profiling report (spreadsheet or doc)

#### 2.6 Desktop Control (Mouse, Keyboard, Window Switching) (Days 50-56, ~15 hours)

**Goal:** Control desktop from gaze + gesture.

- **Mouse Control:**
  - Cross-platform: enigo crate or similar
  - Move cursor to gaze position
  - Click on wink gesture
  - Scroll on finger gesture

- **Gesture Actions:**
  - Pinch → zoom
  - Scroll → scroll
  - Swipe → desktop switch (via AppleScript on macOS, wmctrl on Linux)
  - Palm → Mission Control / App Switcher

- **Error Handling:**
  - Failsafe: Hold ESC to disable
  - Graceful degradation on permission denial (macOS accessibility)

**Deliverables:**
- `src/control/mod.rs` — Control trait
- `src/control/desktop.rs` — Platform-specific implementations
- `src/control/permissions.rs` — macOS accessibility setup helper
- Integration tests (manual for now)

#### 2.7 Performance Tuning + Optimization (Days 57-63, ~20 hours)

**Goal:** Hit 60 FPS on target hardware.

- **Profiling:**
  - Use `flamegraph`, `perf` on Linux
  - Instruments on macOS
  - Profile memory usage
  
- **Optimization strategies:**
  - Memory pooling (pre-allocate all tensors)
  - SIMD for landmark preprocessing (normalize, distance calculations)
  - Parallelize inference (rayon for batch processing if applicable)
  - Reduce allocations in hot loops
  - GPU batching if multiple frames queued

- **Benchmarking:**
  - Measure latency by stage
  - FPS under load (30 min sustained run)
  - Memory stability (no leaks)

**Deliverables:**
- Flamegraph results + analysis
- Performance report (latency breakdown, FPS chart)
- Optimization commit history
- `benches/` suite with detailed measurements

---

### Phase 3: Polish + Deployment (Weeks 13-16, ~100 hours)

#### 3.1 CLI + Configuration (Week 13, ~20 hours)
- `clap` for argument parsing
- Config file format (TOML): model paths, calibration, sensitivity settings
- Help text, examples
- Environment variable overrides

**Deliverables:**
- `src/cli.rs` — Argument parsing
- `config.toml.example` — Example config
- `docs/CONFIGURATION.md` — All tunable parameters

#### 3.2 Calibration UI (Week 13, ~15 hours)
- 9-point calibration (display grid, user clicks/winks at each point)
- Save homography to file
- Auto-load on startup
- Re-calibration without restart

**Deliverables:**
- `src/calibration.rs` — Calibration logic
- `src/calibration_ui.rs` — Terminal or minimal UI

#### 3.3 Logging + Observability (Week 14, ~15 hours)
- `log` crate with different levels
- Frame-by-frame logging (optional debug mode)
- Latency histogram output
- Gesture trigger logs

**Deliverables:**
- Logging throughout codebase
- `docs/DEBUG_LOG_FORMAT.md` — How to interpret logs

#### 3.4 Testing Suite (Week 14, ~20 hours)
- Unit tests for all modules
- Integration tests for full pipeline
- Benchmark suite (criterion)
- CI/CD setup for automated testing

**Deliverables:**
- 50+ tests with >80% code coverage
- `tests/` directory fully populated
- GitHub Actions workflow

#### 3.5 Documentation (Week 15, ~20 hours)
- Architecture doc (system design, data flow)
- Build + deployment instructions (macOS, Linux)
- API docs (rustdoc comments)
- Blog post: "Building Real-Time ML in Rust" (portfolio piece)

**Deliverables:**
- `README.md` — Quick start
- `docs/ARCHITECTURE.md` — System design
- `docs/BUILD.md` — Build instructions
- `docs/DEPLOYMENT.md` — Production setup
- Blog post draft

#### 3.6 Cross-Platform Testing + Hardening (Week 15-16, ~15 hours)
- Test on M4 Mac (development)
- Test on PC with RTX 3070 Ti (training, inference)
- Test on Linux if possible (CI)
- Handle platform differences gracefully

**Deliverables:**
- Platform compatibility matrix
- Known issues + workarounds doc

#### 3.7 Release + Portfolio Polish (Week 16, ~10 hours)
- Tag version `v1.0.0`
- Create GitHub release with binary attachments
- Update README with before/after metrics
- Polish repository (cleanup old code, organize docs)

**Deliverables:**
- GitHub release page
- Final README with metrics
- Portfolio link ready

---

## Learning Objectives

### Rust Systems Programming
- [ ] Master ownership and borrowing in a multi-threaded system
- [ ] Write unsafe Rust for memory-critical paths (frame buffers, inference tensors)
- [ ] FFI with C libraries (MediaPipe, ONNX Runtime, system APIs)
- [ ] Profiling and optimization with flamegraph, perf
- [ ] Error handling in real-time systems (recovery, graceful degradation)
- [ ] Cross-platform conditional compilation (#[cfg])

### Machine Learning
- [ ] Dataset collection and labeling (real-world ML work)
- [ ] Exploratory data analysis and visualization
- [ ] Training regression (gaze) and classification (gesture) models
- [ ] LSTM/RNN for sequence modeling
- [ ] Model export and deployment (ONNX)
- [ ] Inference optimization (batching, precision, latency)
- [ ] Validation metrics and error analysis

### Systems Integration
- [ ] Webcam drivers and frame capture
- [ ] Desktop APIs (mouse, keyboard, window management)
- [ ] Real-time event loops and timing constraints
- [ ] Memory profiling and optimization
- [ ] Performance benchmarking
- [ ] CI/CD and testing strategies

---

## Hardware & Environment

### Development Machine (M4 Mac)
- **Primary use:** Code editor, Rust development, testing, final binary build
- **Setup:** Install Rust via rustup, VS Code + rust-analyzer
- **Python environment:** `python3.10+`, PyTorch (Metal backend)

### Training Machine (PC: Ryzen 9 5900X, RTX 3070 Ti)
- **Primary use:** Model training, inference benchmarking
- **Setup:** CUDA 11.8+, PyTorch with cu118, ONNX Runtime GPU
- **Inference testing:** Validate ONNX models before Rust integration

### Target Environment
- macOS 13.0+ OR Linux (Debian/Ubuntu)
- Webcam + display
- ~200 MB disk space
- Accessibility permissions on macOS (mouse/keyboard control)

---

## Timeline & Milestones

| Week | Phase | Deliverables | Status |
|------|-------|--------------|--------|
| 1 | 1.1 | Data collected, EDA complete | — |
| 2 | 1.3 | Gaze model trained, RMSE <50px | — |
| 3 | 1.4 | Gesture LSTM trained, >95% acc | — |
| 4 | 1.5 | Both models exported to ONNX | — |
| 5 | 2.1 | Rust project scaffolded, dependencies installed | — |
| 6 | 2.2 | Frame capture working, 30 FPS | — |
| 7 | 2.3 | Inference pipeline working, <5ms latency | — |
| 8 | 2.4 | MediaPipe landmark extraction, smoothing | — |
| 9 | 2.5 | Main loop, gesture state machine, <16ms target | — |
| 10 | 2.6 | Desktop control (mouse, gestures) | — |
| 11 | 2.7 | Optimization pass, 60 FPS achieved | — |
| 12 | 2.7 | Performance benchmarks finalized | — |
| 13 | 3.1-3.2 | CLI, config, calibration | — |
| 14 | 3.3-3.4 | Logging, testing suite | — |
| 15 | 3.5-3.6 | Documentation, cross-platform testing | — |
| 16 | 3.7 | Release, portfolio polish | — |

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MediaPipe inference in Rust is complex | Medium | High | Start with ONNX Runtime (simpler), fallback to Python subprocess |
| CUDA/GPU issues on RTX 3070 Ti | Low | High | Test PyTorch + ONNX GPU early (Week 1) |
| Data collection takes longer than expected | Medium | Medium | Automate labeling with existing Python app, use synthetic augmentation |
| 60 FPS target unrealistic on M4 | Low | Medium | Set fallback: 40 FPS acceptable, profile to find bottleneck |
| Rust FFI complexity | Medium | Medium | Budget extra time for FFI debugging; use existing crates when possible |
| Model overfitting to personal data | Medium | Low | Use 70/15/15 split, augmentation, validation on held-out data |

---

## Success Criteria

### Technical
- [ ] Standalone Rust binary runs on M4 and PC
- [ ] 60+ FPS sustained for 30+ minutes
- [ ] Gaze error <50 px, gesture accuracy >95%
- [ ] <16 ms latency per frame (measured)
- [ ] All tests passing on CI

### Portfolio
- [ ] GitHub repo with clear README, architecture doc, and build instructions
- [ ] Blog post explaining the project, learning outcomes, and performance
- [ ] Trained models + ONNX exports included
- [ ] Before/after metrics (Python vs Rust, CPU vs GPU)

### Learning
- [ ] Comfortable with unsafe Rust, memory management, FFI
- [ ] Understand ML pipeline: data → train → export → deploy
- [ ] Can profile and optimize Rust code for latency/throughput

---

## Next Steps

1. **This week:** Set up PyTorch on PC, start data collection script
2. **Week 2:** Collect gaze dataset (30-60 min recording)
3. **Week 3:** Train and validate gaze model
4. **Week 4:** Collect gesture data, train LSTM, export both models
5. **Week 5:** Create Rust project scaffold, set up CI
6. **Week 6:** Implement frame capture, validate 30 FPS

**Decision point after Week 4:** Review trained models. If metrics don't meet targets (gaze <50px, gesture >95%), iterate or augment data before moving to Rust.
