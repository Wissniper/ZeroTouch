# Getting Started: Week 1 Checklist

This document covers the immediate next steps to kick off the C++ + ML rewrite project.

---

## Pre-Project Setup (Before Week 1)

### Hardware Verification

**M4 Mac (Development):**
```bash
# Verify Python + PyTorch
python3 --version  # Should be 3.10+
pip install torch torchvision  # Install PyTorch with Metal GPU support

# Check Metal GPU availability
python3 -c "import torch; print(torch.backends.mps.is_available())"
# Expected: True
```

**Desktop PC (Ryzen 5900X + RTX 3070 Ti):**
```bash
# Verify CUDA + PyTorch
nvidia-smi  # Check GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Check CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"
# Expected: True
```

### Directory Structure

Create the project directories:

```bash
cd ~/Github/Personal/ZeroTouch

# Create data collection directory
mkdir -p data/{raw,processed,models}

# Create scripts directory
mkdir -p scripts/{data_collection,training,validation}

# Create C++ project structure (will do in Week 5)
# mkdir irisflow-cpp

# Ensure .planning directory exists
mkdir -p .planning
```

---

## Week 1: Data Collection & Setup

**Goal:** Collect initial datasets and set up Python environment for training.

### 1.1 Create Python Environment (Days 1-2)

```bash
# Create virtual environment
python3 -m venv venv_ml
source venv_ml/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # GPU version
pip install numpy pandas matplotlib opencv-python scipy scikit-learn jupyter tqdm

# Verify installation
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

### 1.2 Create Data Collection Scripts (Days 2-3)

Create `scripts/data_collection/collect_gaze.py`:
```python
#!/usr/bin/env python3
"""Gaze data collection script."""

import cv2
import json
import time
from pathlib import Path
from src.core.tracker import FaceTracker

# See PLANNING/01_DATA_COLLECTION_GUIDE.md for full implementation
# This is a placeholder - fill in from the guide document

def main():
    print("Gaze data collection starting...")
    # Implementation here
    pass

if __name__ == "__main__":
    main()
```

Create `scripts/data_collection/collect_gestures.py`:
```python
#!/usr/bin/env python3
"""Gesture data collection script."""

# See PLANNING/01_DATA_COLLECTION_GUIDE.md for full implementation

def main():
    print("Gesture data collection starting...")
    # Implementation here
    pass

if __name__ == "__main__":
    main()
```

**Deliverables by end of Day 3:**
- [ ] Two runnable collection scripts
- [ ] Test that they load MediaPipe successfully
- [ ] Test camera capture works

### 1.3 Collect Gaze Dataset (Days 4-5)

```bash
source venv_ml/bin/activate
python scripts/data_collection/collect_gaze.py
```

**During collection:**
- Keep consistent lighting (no backlit scenes)
- Position camera at eye level
- Record 50+ rounds of 9-point grid (target: 5000+ samples)
- Take breaks every 15 minutes

**Expected output:**
- `data/raw/gaze_dataset.json` (3-5 MB, 5000+ samples)

### 1.4 Validate Gaze Data (Day 6)

Create `scripts/validation/validate_gaze.py`:
```python
#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load dataset
    with open("data/raw/gaze_dataset.json") as f:
        samples = json.load(f)
    
    print(f"Loaded {len(samples)} samples")
    
    # Check coverage
    xs = [s['screen_x'] for s in samples]
    ys = [s['screen_y'] for s in samples]
    
    print(f"X range: {min(xs)} - {max(xs)}")
    print(f"Y range: {min(ys)} - {max(ys)}")
    
    # Visualize
    plt.scatter(xs, ys, alpha=0.1, s=1)
    plt.xlabel("Screen X")
    plt.ylabel("Screen Y")
    plt.title(f"Gaze Distribution ({len(samples)} samples)")
    plt.savefig("data/processed/gaze_distribution.png")
    print("Saved visualization to data/processed/gaze_distribution.png")

if __name__ == "__main__":
    main()
```

Run validation:
```bash
python scripts/validation/validate_gaze.py
```

**Check:**
- [ ] Data covers full screen (X: 0-1920, Y: 0-1080)
- [ ] No NaN or Inf values
- [ ] Distribution looks uniform
- [ ] Metadata present (yaw, pitch, pose variations)

### 1.5 Collect Gesture Dataset (Day 7)

```bash
python scripts/data_collection/collect_gestures.py
```

**During collection:**
- Perform each gesture for 30 seconds (idle: 60 seconds)
- Vary hand scale, angle, lighting
- Record at least 6 samples per gesture type

**Expected output:**
- `data/raw/gesture_sequences.pkl` (2-3 MB, 2000+ sequences)

### 1.6 Validate Gesture Data (Day 7, end of day)

Create `scripts/validation/validate_gestures.py`:
```python
#!/usr/bin/env python3
import pickle
import matplotlib.pyplot as plt

def main():
    # Load dataset
    with open("data/raw/gesture_sequences.pkl", "rb") as f:
        sequences = pickle.load(f)
    
    print(f"Loaded {len(sequences)} sequences")
    
    # Check distribution
    labels = {}
    for seq in sequences:
        label = seq['label']
        labels[label] = labels.get(label, 0) + 1
    
    print("Label distribution:")
    for label, count in sorted(labels.items()):
        print(f"  {label}: {count}")
    
    # Visualize
    plt.bar(labels.keys(), labels.values())
    plt.xlabel("Gesture")
    plt.ylabel("Count")
    plt.title(f"Gesture Distribution ({len(sequences)} sequences)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("data/processed/gesture_distribution.png")
    print("Saved visualization to data/processed/gesture_distribution.png")

if __name__ == "__main__":
    main()
```

Run validation:
```bash
python scripts/validation/validate_gestures.py
```

**Check:**
- [ ] All 6 gesture types present
- [ ] Balanced labels (no single gesture <100 samples)
- [ ] All sequences have 10 frames
- [ ] No NaN landmarks

---

## Week 2: Model Training (Preview)

**Note:** Full training happens in Weeks 2-4. This is just preparation.

### 2.1 Create Jupyter Notebooks

Create `notebooks/01_gaze_training.ipynb`:
- [ ] Load gaze dataset
- [ ] Exploratory analysis
- [ ] Baseline model (linear regression)
- [ ] MLP model
- [ ] Training loop
- [ ] Validation curves
- [ ] Export to ONNX

Create `notebooks/02_gesture_training.ipynb`:
- [ ] Load gesture sequences
- [ ] Data augmentation
- [ ] LSTM model definition
- [ ] Training loop
- [ ] Confusion matrix
- [ ] Export to ONNX

### 2.2 Baseline Models

Quick script to establish RMSE/accuracy baselines:

```python
# scripts/training/baseline_gaze.py
# Linear regression on iris ratio → screen coordinates
# Target: Establish baseline RMSE (e.g., ±80px)

# scripts/training/baseline_gesture.py
# Simple MLP on averaged landmarks
# Target: Establish baseline accuracy (e.g., 75%)
```

---

## Documentation: Week 1

By end of Week 1, create:

### Data Collection Report (`data/COLLECTION_NOTES.md`)

```markdown
# Data Collection Report

## Gaze Dataset
- **Samples:** 5000+
- **Collection date:** 2026-04-17 to 2026-04-21
- **Environment:** Consistent lighting, M4 Mac, 27" display
- **Head poses:** Center, left (±15°), right (±15°), up/down (±10°)
- **Issues encountered:** None
- **Data quality:** Visual check passed, no NaNs

## Gesture Dataset
- **Sequences:** 2000+
- **Gestures:** open_hand, pinch, scroll_ready, swipe, palm, idle
- **Per-gesture samples:** >150 (balanced)
- **Collection date:** 2026-04-22 to 2026-04-23
- **Hand scales:** Far, normal, close
- **Issues:** Idle gesture harder to capture distinctly
- **Data quality:** All sequences 10 frames, no NaN

## Next Steps
1. Week 2: Train gaze regression model
2. Week 3: Train gesture LSTM
3. Week 4: Export both to ONNX, validate
```

### Project Status Checklist (`PLANNING/WEEK1_CHECKLIST.md`)

```markdown
## Week 1 Completion Checklist

### Setup
- [ ] Python environment created and tested
- [ ] PyTorch installed (GPU verified)
- [ ] MediaPipe models downloaded
- [ ] Directory structure created

### Data Collection
- [ ] Gaze collection script written
- [ ] Gaze dataset collected (5000+ samples)
- [ ] Gaze data validated (no NaNs, full coverage)
- [ ] Gesture collection script written
- [ ] Gesture dataset collected (2000+ sequences)
- [ ] Gesture data validated (balanced labels)

### Documentation
- [ ] Collection methodology documented
- [ ] Data quality report written
- [ ] Next steps (training) outlined

### Learning
- [ ] Understand data collection challenges
- [ ] Familiar with dataset formats (JSON, pickle)
- [ ] Can visualize and validate data

### Status
**Ready for Week 2: Model Training**
```

---

## Quick Decision Tree: "What If..."

### What if data collection takes longer?
- **Plan:** Collect in parallel (gaze Day 1-5, gestures Day 6-7)
- **Fallback:** Continue into Week 1.5; adjust timeline
- **Minimum viable:** 2000 gaze samples, 1000 gesture sequences

### What if PyTorch/CUDA setup fails?
- **Local GPU (RTX 3070 Ti):** Primary training machine
- **Fallback:** Google Colab (free, GPU access)
- **Last resort:** CPU training (slow but works for small models)

### What if camera/MediaPipe isn't detecting?
- **Check:** Camera permissions, lighting, distance from camera
- **Debug:** Run `cv2.imshow()` to visualize frame
- **Fallback:** Temporarily use synthetic data for training

### What if gesture data is imbalanced?
- **Solution:** Oversample minority classes or use weighted loss
- **Collect more:** Idle gesture takes longest; record extra

---

## Timeline: Strict Path

| Date | Day | Milestone | Status |
|------|-----|-----------|--------|
| 2026-04-17 | 1 | Python setup, start gaze collection | — |
| 2026-04-18 | 2 | Gaze collection (continuous) | — |
| 2026-04-19 | 3 | Gaze collection (continuous) | — |
| 2026-04-20 | 4 | Gaze collection (continuous) | — |
| 2026-04-21 | 5 | Gaze collection finish, validation | — |
| 2026-04-22 | 6 | Gesture collection (Day 1) | — |
| 2026-04-23 | 7 | Gesture collection (Day 2), validation | **Week 1 DONE** |

---

## Success Criteria for Week 1

**GO to Week 2 if:**
- [ ] 5000+ gaze samples collected and validated
- [ ] 2000+ gesture sequences collected and validated
- [ ] No data quality issues (NaNs, incomplete landmarks, etc.)
- [ ] Comfortable with collection methodology
- [ ] Understand dataset structure and format

**REDO if:**
- [ ] Data has >5% NaN values
- [ ] Gaze coverage <60% of screen area
- [ ] Gesture labels severely imbalanced (one gesture <100 samples)
- [ ] Collection notes don't match data (documentation mismatch)

---

## Expected Artifacts at End of Week 1

```
data/
├── raw/
│   ├── gaze_dataset.json         (5000+ samples, 3-5 MB)
│   └── gesture_sequences.pkl     (2000+ sequences, 2-3 MB)
├── processed/
│   ├── gaze_distribution.png     (visualization)
│   └── gesture_distribution.png  (visualization)
└── COLLECTION_NOTES.md           (methodology, quality report)

scripts/
├── data_collection/
│   ├── collect_gaze.py
│   └── collect_gestures.py
└── validation/
    ├── validate_gaze.py
    └── validate_gestures.py

PLANNING/
├── 00_GETTING_STARTED.md         (this file)
├── WEEK1_CHECKLIST.md            (completion status)
├── CPP_ML_REWRITE_PLAN.md       (master plan)
├── 01_DATA_COLLECTION_GUIDE.md   (detailed methodology)
├── 02_CPP_ARCHITECTURE.md       (system design)
└── 03_LEARNING_OBJECTIVES.md     (skills & progress)
```

---

## Commands to Run This Week

**Day 1-2 Setup:**
```bash
cd ~/Github/Personal/ZeroTouch

# Create venv
python3 -m venv venv_ml
source venv_ml/bin/activate

# Install dependencies
pip install torch numpy pandas matplotlib opencv-python scipy scikit-learn

# Verify
python3 -c "import torch; print('OK')"
```

**Day 3+ Data Collection:**
```bash
source venv_ml/bin/activate

# Gaze collection
python scripts/data_collection/collect_gaze.py

# Gesture collection (after gaze done)
python scripts/data_collection/collect_gestures.py

# Validation
python scripts/validation/validate_gaze.py
python scripts/validation/validate_gestures.py
```

---

## Key Contacts & Resources (Week 1)

**If stuck on:**
- PyTorch/CUDA: Check official docs + StackOverflow
- MediaPipe issues: Check MediaPipe GitHub issues
- Data quality: Review PLANNING/01_DATA_COLLECTION_GUIDE.md
- Architecture questions: Review PLANNING/CPP_ML_REWRITE_PLAN.md

---

## Next: Week 2

Once Week 1 is complete, move to **Phase 1.3-1.5: Model Training**

Preview:
- [ ] Train gaze regression model (Week 2)
- [ ] Train gesture LSTM (Week 3)
- [ ] Export both to ONNX (Week 4)
- [ ] C++ development begins (Week 5)

See `PLANNING/CPP_ML_REWRITE_PLAN.md` for detailed timeline.
