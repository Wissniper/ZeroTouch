# Learning Objectives & Progress Tracker

This document tracks your learning goals across **Rust systems programming** and **Machine Learning**, with checkpoints at each phase.

---

## Rust Systems Programming Learning Path

### Module 1: Memory & Ownership (Weeks 1-2)

**Concepts to master:**
- [ ] Rust ownership model: move semantics, borrowing, lifetimes
- [ ] Stack vs. heap allocation
- [ ] Smart pointers: `Box`, `Rc`, `Arc`, `RefCell`, `Mutex`
- [ ] Lifetime annotations and variance
- [ ] Practical: Avoid borrow checker errors in the ring buffer

**Checkpoint Exercise (Week 2):**
> Implement a generic circular ring buffer that stores pre-allocated frames without copying.
> - Use `unsafe` blocks only where necessary (pointer arithmetic)
> - Document safety invariants
> - Write test that verifies wraparound correctness

**Resources:**
- Rust Book Ch. 4 (Ownership) + Ch. 19 (Advanced Features, unsafe)
- Watch: Jon Gjengset's "Crust of Rust" (5-part series on ownership)

---

### Module 2: Unsafe Rust & FFI (Weeks 3-4)

**Concepts to master:**
- [ ] Unsafe Rust: When to use it, why it's safe in context
- [ ] Foreign Function Interface (FFI): Calling C libraries from Rust
- [ ] Pointer manipulation: `*const`, `*mut`, `std::ptr`
- [ ] Memory layout: `#[repr(C)]`, `std::mem::transmute` (when safe)
- [ ] ONNX Runtime Rust bindings: Using existing crates like `ort`

**Checkpoint Exercise (Week 4):**
> Wrap ONNX Runtime FFI calls in a safe Rust interface.
> - Load a model (`gaze.onnx`)
> - Run inference on dummy input
> - Verify output shape and values
> - Handle errors gracefully (missing model, shape mismatch)

**Resources:**
- Rust Book Ch. 19.1 (Unsafe Rust)
- "The Rustonomicon" (FFI, memory layout)
- tch-rs / ort crate documentation

---

### Module 3: Concurrent & Async Rust (Weeks 5-6)

**Concepts to master:**
- [ ] Threading: `std::thread`, data sharing with `Arc<Mutex<T>>`
- [ ] Async/await: `tokio`, futures, pinning
- [ ] Channels: `mpsc` for thread communication
- [ ] Race conditions, deadlock prevention, `Send` + `Sync` traits
- [ ] Practical: Separate camera capture thread from inference thread (optional)

**Checkpoint Exercise (Week 6):**
> Build a simple producer-consumer pipeline:
> - Thread A: Read frames from camera, send via channel
> - Thread B: Receive frames, run mock inference
> - Measure throughput (frames/sec) and latency
> - Safely share state across threads (no data races)

**Resources:**
- Rust Book Ch. 16 (Fearless Concurrency)
- Tokio tutorial: https://tokio.rs
> Practical book: "Concurrency in Rust" by Bear (O'Reilly)

---

### Module 4: Performance & Profiling (Weeks 7-8)

**Concepts to master:**
- [ ] Benchmark design: `criterion` crate
- [ ] Profiling tools: `flamegraph`, `perf` (Linux), Instruments (macOS)
- [ ] Memory profiling: `valgrind`, Rust's `leak` detection
- [ ] SIMD optimization: `packed_simd`, intrinsics (optional)
- [ ] Identifying bottlenecks: CPU-bound vs. I/O-bound vs. allocation-bound

**Checkpoint Exercise (Week 8):**
> Profile the gaze inference pipeline:
> - Measure per-frame latency breakdown (detection, inference, smoothing)
> - Generate flamegraph showing time spent in each module
> - Identify top 3 bottlenecks
> - Propose 2-3 optimization ideas
> - Implement one optimization and measure improvement

**Resources:**
- Criterion benchmarking guide: https://bheisler.github.io/criterion.rs/book
- Flamegraph: https://www.brendangregg.com/flamegraphs.html
- `perf` tutorial

---

### Module 5: Systems Programming: Low-Level APIs (Weeks 9-10)

**Concepts to master:**
- [ ] Platform-specific APIs: macOS AVFoundation, Linux v4l2
- [ ] System calls: `ioctl`, file descriptors, mmap (for zero-copy)
- [ ] Error codes: POSIX errno, graceful handling
- [ ] Conditional compilation: `#[cfg]` for macOS vs. Linux code paths

**Checkpoint Exercise (Week 10):**
> Implement cross-platform frame capture:
> - macOS: Use AVFoundation to capture frames
> - Linux: Use v4l2-rs to capture from `/dev/video0`
> - Measure frame rate and latency on both platforms
> - Handle camera disconnection gracefully

**Resources:**
- AVFoundation docs: https://developer.apple.com/documentation/avfoundation
- Video4Linux docs: https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/

---

### Module 6: System Integration & Deployment (Weeks 11-12)

**Concepts to master:**
- [ ] CLI argument parsing: `clap` crate
- [ ] Configuration files: TOML/YAML parsing
- [ ] Logging: `log` + `env_logger` or `tracing` crate
- [ ] Signal handling: `SIGTERM`, graceful shutdown
- [ ] Cross-platform distribution: Cargo release, binary packaging

**Checkpoint Exercise (Week 12):**
> Create a production-ready binary:
> - CLI with `--help`, `--config`, `--log-level`
> - Config file support (TOML)
> - Graceful shutdown on SIGTERM
> - Build and test on macOS and Linux
> - Create GitHub release with pre-built binaries

**Resources:**
- `clap` documentation: https://docs.rs/clap/
- `cargo-dist`: https://rust-lang.github.io/cargo-dist/

---

## Machine Learning Learning Path

### Phase 1: Dataset & Exploration (Weeks 1-2)

**Concepts to master:**
- [ ] Data collection best practices (reproducibility, quality checks)
- [ ] CSV/JSON/pickle file formats
- [ ] Exploratory Data Analysis (EDA): visualizations, distributions
- [ ] Train/validation/test splits
- [ ] Data augmentation strategies

**Checkpoint Exercise (Week 2):**
> Collect and validate your gaze dataset:
> - Record 5000+ gaze samples across full screen
> - Create visualization: scatter plot of gaze positions
> - Compute statistics: coverage, outliers, missing values
> - Document collection methodology (lighting, head poses, etc.)

**References:**
- Kaggle: EDA tutorials
- "Hands-On Machine Learning" Ch. 2 (Project Setup)

---

### Phase 2: Regression (Gaze Estimation) (Weeks 2-3)

**Concepts to master:**
- [ ] Supervised learning: regression vs. classification
- [ ] Loss functions: MSE, MAE, Huber
- [ ] Model evaluation metrics: RMSE, MAE, R²
- [ ] Overfitting: validation curves, early stopping
- [ ] PyTorch fundamentals: tensors, autograd, training loops

**Checkpoint Exercise (Week 3):**
> Train a gaze regression model:
> - Baseline: Linear regression on raw iris ratio (establish RMSE)
> - Simple MLP: 468 face landmarks → (x, y) screen coords
> - Plot training/validation curves
> - Evaluate on test set: <50px RMSE
> - Analyze failure cases: where does model struggle?

**PyTorch Code Example:**
```python
import torch
import torch.nn as nn

class GazeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(468*3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)  # x, y
    
    def forward(self, landmarks):
        x = torch.relu(self.fc1(landmarks))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Training loop
model = GazeModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(100):
    for batch, (landmarks, labels) in enumerate(train_loader):
        pred = model(landmarks)
        loss = criterion(pred, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    val_loss = evaluate(model, val_loader)
    print(f"Epoch {epoch}: train_loss={loss:.4f}, val_loss={val_loss:.4f}")
```

**References:**
- PyTorch tutorials: https://pytorch.org/tutorials
- "Deep Learning" Ch. 6 (Feedforward Networks)

---

### Phase 3: Sequence Modeling (Gesture Recognition) (Weeks 3-4)

**Concepts to master:**
- [ ] Recurrent Neural Networks (RNNs), LSTM, GRU
- [ ] Sequence classification: many-to-one architecture
- [ ] Data augmentation for sequences (time warping, jitter)
- [ ] Class imbalance: weighted loss, stratified sampling
- [ ] Confusion matrix, per-class metrics

**Checkpoint Exercise (Week 4):**
> Train a gesture recognition LSTM:
> - Dataset: 2000+ 10-frame gesture sequences
> - Model: LSTM (630 dims) → hidden (128) → softmax (6 classes)
> - Metrics: >95% accuracy, per-class precision/recall
> - Data augmentation: time warping, landmark jitter
> - Analyze confusion matrix: which gestures confuse the model?

**PyTorch LSTM Code:**
```python
class GestureRecognitionLSTM(nn.Module):
    def __init__(self, input_size=630, hidden_size=128, num_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, sequences):
        # sequences: [batch_size, 10, 630]
        lstm_out, (h_n, c_n) = self.lstm(sequences)
        # Use last hidden state
        last_hidden = h_n[-1]  # [batch_size, 128]
        logits = self.fc(last_hidden)  # [batch_size, 6]
        return logits

# Training with class weights
class_weights = torch.tensor([1.0, 1.0, 1.2, 1.1, 0.9, 0.8])
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**References:**
- PyTorch LSTM documentation
- "Sequence Models" (Andrew Ng): https://www.deeplearning.ai
- Chris Olah's blog on RNNs: http://colah.github.io/posts/2015-08-Understanding-LSTMs/

---

### Phase 4: Model Deployment & Export (Week 4)

**Concepts to master:**
- [ ] Model export: PyTorch → ONNX
- [ ] ONNX Runtime: CPU vs. GPU inference
> [ ] Quantization: FP32 → INT8 (optional, for speed)
- [ ] Inference validation: outputs match PyTorch exactly

**Checkpoint Exercise (Week 4):**
> Export and validate both models:
> - Convert gaze model to ONNX
> - Convert gesture LSTM to ONNX
> - Load both in ONNX Runtime (CPU + GPU)
> - Validate outputs match PyTorch (tolerance: <1e-5)
> - Measure inference latency: CPU vs. GPU

**Export Code:**
```python
import torch.onnx

# Export gaze model
dummy_input = torch.randn(1, 468*3)
torch.onnx.export(
    gaze_model,
    dummy_input,
    "models/gaze.onnx",
    input_names=["landmarks"],
    output_names=["x", "y"],
    opset_version=11,
)

# Verify with ONNX Runtime
import onnxruntime as ort

session = ort.InferenceSession("models/gaze.onnx", providers=['CUDAExecutionProvider'])
output = session.run(None, {"landmarks": dummy_input.numpy()})
```

---

### Phase 5: ML Systems (Data Pipeline) (Weeks 1-4, ongoing)

**Concepts to master:**
- [ ] Data versioning (git-lfs, DVC)
- [ ] Hyperparameter tuning: grid search, random search, Bayesian
- [ ] Model evaluation: cross-validation, ROC curves, calibration
- [ ] MLOps basics: logging experiments (wandb, MLflow)
- [ ] Reproducibility: random seeds, hardware specs

**Checkpoint Exercise:**
> Set up experiment tracking:
> - Log all hyperparameters to `experiments/` directory
> - Record train/val/test metrics
> - Visualize training curves with matplotlib
> - Document: "Model X achieved Y% accuracy with hyperparams Z"

**Code Structure:**
```
experiments/
├── gaze_v1_baseline.json
├── gaze_v2_augmentation.json
├── gaze_v3_kalman_labels.json  # Better labels → better accuracy
└── gesture_v1_lstm.json
```

---

## Systems Integration: Bringing It Together

### Weeks 9-12: Full-Stack Integration

**Key Milestones:**
1. **Week 9:** Rust inference pipeline loads ONNX models, runs gaze inference
2. **Week 10:** Frame capture + landmark preprocessing integrated
3. **Week 11:** Gesture state machine + temporal filtering working
4. **Week 12:** Full pipeline: frame → gaze → gesture → cursor movement

**Learning:** How ML models integrate into real-time systems:
- [ ] Latency budgets per component
- [ ] Memory management during inference
- [ ] Error handling when model fails
- [ ] A/B testing model versions

---

## Comprehensive Skill Matrix

| Skill | Beginner | Intermediate | Expert | Target |
|-------|----------|--------------|--------|--------|
| **Rust** |  |  |  |  |
| Ownership & borrowing |  | ✓ |  | ✓ |
| Unsafe Rust | ✓ |  |  | ✓ |
| FFI / C interop |  | ✓ |  | ✓ |
| Async/await |  | ✓ |  | ✓ |
| Performance profiling |  | ✓ |  | ✓ |
| **Machine Learning** |  |  |  |  |
| Data collection & validation | ✓ |  |  | ✓ |
| Regression modeling |  | ✓ |  | ✓ |
| Sequence modeling (LSTM) |  | ✓ |  | ✓ |
| Model export & deployment |  | ✓ |  | ✓ |
| Hyperparameter tuning |  | ✓ |  | ✓ |
| **Systems** |  |  |  |  |
| Real-time event loops |  | ✓ |  | ✓ |
| Cross-platform development |  | ✓ |  | ✓ |
| System APIs (camera, mouse) |  | ✓ |  | ✓ |
| Latency profiling |  | ✓ |  | ✓ |

---

## Weekly Checkpoint Template

Use this template to track progress and learning:

```markdown
## Week N: [Phase]

### Completed This Week
- [ ] Learning objective 1
- [ ] Learning objective 2

### Code Written
- Module A: X lines
- Module B: Y lines
- Tests: Z lines

### Challenges Encountered
1. [Challenge] → [Solution]

### Metrics
- FPS: X
- Latency: Ym s
- Test coverage: Z%

### Key Learnings
- Insight 1
- Insight 2

### Next Week
- [ ] Objective 1
- [ ] Objective 2
```

---

## Portfolio Artifacts

By end of project, you'll have these portfolio pieces:

1. **GitHub repository:**
   - 3000+ lines of production Rust
   - 1000+ lines of PyTorch + training notebooks
   - Comprehensive documentation
   - CI/CD setup

2. **Blog post:** "Building Real-Time ML in Rust"
   - Why: Learn new skill + document journey
   - Content: Architecture, performance results, tradeoffs
   - Audience: Rust + ML communities

3. **Trained models + metrics:**
   - Gaze regression: <50px RMSE
   - Gesture LSTM: >95% accuracy
   - Performance: 60+ FPS, <16ms latency

4. **Deployment:**
   - Standalone binary (no Python runtime required)
   - macOS + Linux support
   - Installation script or package manager release

---

## Resource Compilation

### Rust
- Official: https://doc.rust-lang.org/
- "The Rust Book": https://doc.rust-lang.org/book/
- "Rustlings": https://github.com/rust-lang/rustlings
- Jon Gjengset (YouTube): Crust of Rust series
- "Concurrency in Rust" (O'Reilly)

### Machine Learning
- "Hands-On Machine Learning" (2nd Ed) — Géron
- "Dive into Deep Learning" (free online): https://d2l.ai/
- Andrew Ng's ML Specialization (Coursera)
- Fast.ai: Practical Deep Learning

### Performance & Systems
- "Systems Performance" (Brendan Gregg)
- Rust Performance Book: https://nnethercote.github.io/perf-book/
- Flamegraphs: https://www.brendangregg.com/flamegraphs.html

### Vision & Gesture Recognition
- MediaPipe documentation: https://mediapipe.dev/
- "Computer Vision: Algorithms and Applications" (Szeliski)
- OpenCV tutorials

---

## Success Definition

**By end of Week 16:**
- [ ] Comfortable writing idiomatic Rust (ownership, lifetimes, traits)
- [ ] Can profile and optimize Rust code for latency
- [ ] Understand FFI and unsafe Rust in practical context
- [ ] Built and deployed ML models (PyTorch → ONNX → Rust inference)
- [ ] Created a portfolio project that showcases both Rust + ML skills
- [ ] Can explain architecture and tradeoffs in technical interviews

