# Learning Objectives & Progress Tracker

This document tracks your learning goals across **C++ systems programming** and **Machine Learning**, with checkpoints at each phase.

---

## C++ Systems Programming Learning Path

### Module 1: Memory Management & RAII (Weeks 1-2)

**Concepts to master:**
- [ ] RAII (Resource Acquisition Is Initialization)
- [ ] Smart pointers: `std::unique_ptr`, `std::shared_ptr`, `std::weak_ptr`
- [ ] Move semantics and R-value references (`std::move`, `&&`)
- [ ] Stack vs. Heap allocation in C++
- [ ] Rule of Three / Rule of Five

**Checkpoint Exercise (Week 2):**
> Implement a generic circular ring buffer that stores pre-allocated frames without copying.
> - Use `std::unique_ptr` for resource management
> - Implement move constructor and move assignment operator
> - Write test that verifies wraparound correctness

**Resources:**
- "Effective Modern C++" by Scott Meyers (Items 18-22)
- C++ Core Guidelines: Resource Management

---

### Module 2: Pointers, Memory & Low-Level APIs (Weeks 3-4)

**Concepts to master:**
- [ ] Raw pointers and pointer arithmetic (`*`, `&`, `->`)
- [ ] C-style arrays vs. `std::array` and `std::vector`
- [ ] Memory layout and alignment (`alignas`, `sizeof`)
- [ ] C-style string manipulation vs. `std::string_view`
- [ ] Calling C libraries from C++ (`extern "C"`)

**Checkpoint Exercise (Week 4):**
> Wrap ONNX Runtime C API in a safe C++ class.
> - Load a model (`gaze.onnx`)
> - Run inference on dummy input using `Ort::Session`
> - Verify output shape and values
> - Handle errors using exceptions or `std::optional`

**Resources:**
- ONNX Runtime C++ API Documentation
- "C++ Primer" Ch. 12 (Dynamic Memory)

---

### Module 3: Concurrency & Multithreading (Weeks 5-6)

**Concepts to master:**
- [ ] Threading: `std::thread`, `std::jthread` (C++20)
- [ ] Synchronization: `std::mutex`, `std::lock_guard`, `std::unique_lock`
- [ ] Communication: `std::condition_variable`, `std::future`, `std::promise`
- [ ] Atomic operations: `std::atomic`
- [ ] Thread pools and task-based parallelism

**Checkpoint Exercise (Week 6):**
> Build a simple producer-consumer pipeline:
> - Thread A: Read frames from camera, push to thread-safe queue
> - Thread B: Pop frames, run mock inference
> - Measure throughput (frames/sec) and latency
> - Use `std::condition_variable` for efficient signaling

**Resources:**
- "C++ Concurrency in Action" by Anthony Williams
- cppreference: Thread support library

---

### Module 4: Build Systems & Tooling (Weeks 7-8)

**Concepts to master:**
- [ ] CMake: `add_executable`, `target_link_libraries`, `find_package`
- [ ] Compiler flags and optimization levels (`-O3`, `-march=native`)
- [ ] Dependency management: `vcpkg` or `Conan`
- [ ] Static vs. Dynamic linking
- [ ] Profiling: `gprof`, `Valgrind`, `Google Benchmark`

**Checkpoint Exercise (Week 8):**
> Create a robust CMake build system for the project:
> - Support for external libraries (OpenCV, ONNX Runtime)
> - Conditional compilation for macOS/Linux
> - Integrated unit tests using `GTest` or `Catch2`
> - Benchmark suite using `Google Benchmark`

**Resources:**
- "Professional CMake: A Practical Guide"
- CMake Documentation: https://cmake.org/documentation/

---

### Module 5: Platform-Specific Systems APIs (Weeks 9-10)

**Concepts to master:**
- [ ] macOS: AVFoundation (Objective-C++ interop)
- [ ] Linux: Video4Linux2 (v4l2) ioctls
- [ ] System calls and file descriptors
- [ ] Error handling with `errno` and exceptions

**Checkpoint Exercise (Week 10):**
> Implement cross-platform frame capture:
> - macOS: Use `.mm` file to bridge Objective-C AVFoundation to C++
> - Linux: Use `ioctl` to interact with `/dev/video0`
> - Measure frame rate and latency on both platforms

**Resources:**
- Apple Developer: AVFoundation Documentation
- Linux Kernel: V4L2 API Guide

---

### Module 6: System Integration & CLI (Weeks 11-12)

**Concepts to master:**
- [ ] CLI argument parsing: `CLI11` or `boost::program_options`
- [ ] Configuration: JSON (`nlohmann/json`) or YAML (`yaml-cpp`)
- [ ] Logging: `spdlog` or `glog`
- [ ] Signal handling: `std::signal` for graceful shutdown

**Checkpoint Exercise (Week 12):**
> Create a production-ready binary:
> - CLI with `--help`, `--config`, `--verbose`
> - Config file support
> - Graceful shutdown on Ctrl+C
> - Logging to both console and file

---

## Machine Learning Learning Path

[... ML Phases 1-3 remain the same ...]

### Phase 4: Model Deployment & Export (Week 4)

**Concepts to master:**
- [ ] Model export: PyTorch → ONNX
- [ ] ONNX Runtime: CPU vs. GPU inference
- [ ] C++ API for ONNX Runtime
- [ ] Inference validation: outputs match PyTorch exactly

**Checkpoint Exercise (Week 4):**
> Export and validate both models:
> - Convert gaze model to ONNX
> - Convert gesture LSTM to ONNX
> - Load both in ONNX Runtime C++ API
> - Validate outputs match PyTorch
> - Measure inference latency

[... Rest of document remains largely consistent but with C++ context ...]
