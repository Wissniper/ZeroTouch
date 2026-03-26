# IrisFlow — Real-Time Head-Compensated Iris Gaze Tracker

[![Tests](https://github.com/Wissniper/Gesture-And-Eye-Track-System/actions/workflows/tests.yml/badge.svg)](https://github.com/Wissniper/Gesture-And-Eye-Track-System/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A **zero-touch** computer interface that replaces your mouse with your eyes and hands. IrisFlow uses real-time iris tracking with head-pose compensation and hand gesture recognition to control your cursor, click, scroll, zoom, drag, and switch desktops — all through a standard webcam.

<p align="center">
  <em>Demo: iris-controlled cursor with wink-to-click and hand gesture actions</em>
</p>

<!-- Replace with actual recording: screen-record yourself using IrisFlow,
     save as demo.gif in the repo root, and uncomment below:
  <p align="center"><img src="demo.gif" width="600" alt="IrisFlow demo"></p>
-->

---

## Features

| Gesture | Action | Input | Status |
|---------|--------|-------|--------|
| Iris movement | Cursor control | Eye gaze (both eyes averaged) | In Progress |
| Left wink | Left click | Eye blink differential | In Progress |
| Right wink | Right click | Eye blink differential | In Progress |
| 1 finger (index) | Scroll | Index finger Y-velocity | In Progress |
| 2 fingers (pinch) | Zoom | Thumb-index distance delta | In Progress |
| 3 fingers | Drag | Hold mouse button | In Progress |
| 4-finger swipe | Switch desktop | Index finger X-velocity | In Progress |
| 5 fingers (palm) | Mission Control | All fingers extended | In Progress |
| 9-point calibration | Gaze → screen mapping | Homography (RANSAC) | In Progress |
| Auto drift correction | Re-centre gaze mapping | Translation offset every 30s | In Progress |

## How It Works

```
┌──────────────┐      ┌───────────────────┐     ┌─────────────────┐
│   Webcam     │────► │  MediaPipe Face   │────►│  Gaze Ratio     │
│  (640×480)   │      │  468 landmarks    │     │ (iris / socket) │
└──────────────┘      │  + blendshapes    │     └────────┬────────┘
                      │  + transform mat  │              │
                      └───────────────────┘              │
                                                         ▼
┌──────────────┐     ┌───────────────────┐     ┌─────────────────┐
│  PyAutoGUI   │◄────│  One-Euro Filter  │◄────│  Head Comp +    │
│  move/click  │     │  (adaptive smooth)│     │  Homography     │
└──────────────┘     └───────────────────┘     └─────────────────┘

┌──────────────┐     ┌───────────────────┐
│  MediaPipe   │────►│  Gesture          │────► scroll / zoom /
│  Hand (21)   │     │  Controller       │      drag / switch
└──────────────┘     └───────────────────┘
```

### Key Algorithms

- **One-Euro Filter** (`src/core/processor.py`): Adaptive low-pass filter that minimises jitter at rest and lag during fast movement. Implements the [Casiez et al. 2012](https://cristal.univ-lille.fr/~casiez/1euro/) algorithm from scratch — no external library.
- **9-Point Homography Calibration** (`src/core/calibration.py`): Maps the non-linear "bowl" of iris-in-eye-socket ratios to flat screen coordinates using `cv2.findHomography` with RANSAC outlier rejection.
- **Head-Pose Compensation** (`src/core/tracker.py`): Extracts yaw/pitch from MediaPipe's facial transformation matrix and subtracts it from the gaze vector, decoupling intentional gaze shifts from head movement.
- **Wink Detection** (`src/gestures/gestures.py`): Uses blendshape score *differential* (left - right) to distinguish intentional winks from blinks, with configurable cooldown to prevent double-triggers.

## Project Structure

```
├── src/
│   ├── main.py              # Entry point, main loop, FPS counter
│   ├── core/
│   │   ├── tracker.py       # MediaPipe face + hand detection wrapper
│   │   ├── processor.py     # One-Euro filter implementation
│   │   └── calibration.py   # 9-point homography + drift correction
│   └── gestures/
│       └── gestures.py      # Wink, pinch, scroll, swipe detectors
├── tests/                   # 59 unit tests (pytest)
├── setup_models.py          # Downloads MediaPipe .task models
├── pyproject.toml           # Package config, deps, entry points
├── Makefile                 # install / run / test / lint / clean
├── REPORT.md                # Technical write-up and design decisions
└── .github/workflows/       # CI: tests on Python 3.10-3.12
```

## Quick Start

### Prerequisites

- Python 3.10+
- A webcam
- macOS / Linux (Windows: untested but should work)

### Installation

```bash
# Clone
git clone https://github.com/Wissniper/Gesture-And-Eye-Track-System.git
cd Gesture-And-Eye-Track-System

# Create venv and install
python -m venv .venv
source .venv/bin/activate
make install        # or: pip install -e .

# Download MediaPipe models (~11 MB)
make setup          # or: python setup_models.py
```

### Run

```bash
make run            # or: python -m src
```

1. **Calibration**: Look at each of the 9 red dots and press SPACE.
2. **Use**: Move your eyes to control the cursor. Wink to click. Show hand gestures for scroll/zoom/drag.
3. **Exit**: Press ESC.

### Run Tests

```bash
make install-dev    # install pytest, black, mypy
make test           # 59 tests with coverage report
```

## Configuration

Key parameters in `src/main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HEAD_COMP_SCALE` | 0.012 | Head-pose compensation strength |
| `SCROLL_SENSITIVITY` | 15 | Scroll clicks per gesture frame |
| `DRIFT_INTERVAL_S` | 30 | Seconds between auto drift corrections |

Gesture tuning in `GestureController.__init__()`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wink_threshold` | 0.06 | Blink score differential for wink detection |
| `cooldown_ms` | 500 | Minimum ms between repeated gesture triggers |

One-Euro filter tuning in `GazeProcessor.__init__()`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_cutoff` | 0.8 | Lower = smoother at rest, more lag |
| `beta` | 0.02 | Higher = less lag during fast moves, more jitter |

## Technical Details

See [REPORT.md](REPORT.md) for an in-depth discussion of:
- The iris-to-screen coordinate mapping problem
- Head compensation mathematics
- Zone-velocity design (inner/middle/outer)
- Smoothing algorithm comparison (EMA vs One-Euro vs Moving Median)
- Bugs found and fixed during development
- Future improvement roadmap

## License

[MIT](LICENSE) — Wisdom Ononiba, 2026
