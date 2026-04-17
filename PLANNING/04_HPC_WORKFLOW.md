# UGent HPC Integration: ML Training Workflow

This document outlines how to use UGent's HPC cluster for fast model training while developing Rust locally.

---

## Workflow Overview

```
LOCAL (M4 Mac)              HPC (UGent Cluster)
─────────────────          ──────────────────

Week 1: Data Collection
├─ Collect gaze data
└─ Collect gesture data
    │
    └─ Upload to HPC
         │
    Week 2-4: Model Training (HPC)
    ├─ 01_gaze_training.ipynb (Jupyter)
    ├─ 02_gesture_training.ipynb (Jupyter)
    └─ Export ONNX models
         │
         └─ Download models to M4
              │
Week 5+: Rust Development (M4)
├─ Build Rust binary
├─ Load ONNX models
└─ Test on local machine
```

---

## Prerequisites

Before starting, verify HPC access:

```bash
# Login to HPC
ssh your_username@login.hpc.ugent.be

# Check GPU availability
module avail CUDA
module avail PyTorch

# Check storage quota
quota

# List available GPUs
# (command varies by cluster; ask HPC admins)
```

---

## Week 1: Data Collection (Local M4)

Same as in `00_GETTING_STARTED.md`:

1. Collect gaze dataset (5000+ samples)
2. Collect gesture dataset (2000+ sequences)
3. Validate both

**Output:**
```
data/raw/
├── gaze_dataset.json        (~5 MB)
└── gesture_sequences.pkl    (~2 MB)
```

---

## Upload Datasets to HPC

After Week 1 validation, upload to HPC storage:

### Option A: Using `scp` (Simple Copy)

```bash
# From your M4, upload data to HPC
scp data/raw/gaze_dataset.json \
    your_username@login.hpc.ugent.be:/home/your_username/irisflow/data/

scp data/raw/gesture_sequences.pkl \
    your_username@login.hpc.ugent.be:/home/your_username/irisflow/data/
```

### Option B: Using `rsync` (Better for Large Files)

```bash
# Faster, with progress bar and resume capability
rsync -avz --progress data/raw/ \
    your_username@login.hpc.ugent.be:/home/your_username/irisflow/data/raw/
```

### Option C: Git-LFS (Recommended if Already Using Git)

```bash
# Install git-lfs on M4
brew install git-lfs
git lfs install

# Track data files
git lfs track "data/raw/*.json"
git lfs track "data/raw/*.pkl"

# Commit and push
git add data/raw/
git commit -m "Add training datasets"
git push origin main

# On HPC: clone and pull
cd /home/your_username/irisflow
git clone <your_repo>
git lfs pull
```

---

## Week 2-4: Model Training on HPC

### HPC Directory Structure

```
/home/your_username/irisflow/
├── data/
│   ├── raw/
│   │   ├── gaze_dataset.json
│   │   └── gesture_sequences.pkl
│   └── processed/
├── notebooks/
│   ├── 01_gaze_training.ipynb
│   └── 02_gesture_training.ipynb
├── models/
│   ├── gaze.pt              (PyTorch checkpoint)
│   ├── gesture.pt
│   ├── gaze.onnx            (Exported ONNX)
│   └── gesture.onnx
└── requirements.txt
```

### Setup Python Environment on HPC

**SSH into HPC:**
```bash
ssh your_username@login.hpc.ugent.be
```

**Load modules:**
```bash
# Check available modules
module avail

# Load Python + CUDA (example; adjust to your HPC)
module load Python/3.10.13-GCCcore-13.2.0
module load CUDA/12.2.0
module load cuDNN/8.9.1.23-CUDA-12.2.0

# Verify
python --version
nvcc --version
```

**Create virtual environment:**
```bash
cd ~/irisflow
python -m venv venv_ml
source venv_ml/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install jupyter numpy pandas matplotlib scikit-learn opencv-python scipy tqdm

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name()}')"
```

### Run Jupyter Notebook on HPC

**Option 1: Interactive Session (Quick Testing)**

```bash
# Request interactive GPU node
salloc -N 1 -t 01:00:00 --gres=gpu:1 --partition=gpu
# (exact command depends on your HPC; ask admin)

# Start Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888

# From your M4, create SSH tunnel
ssh -L 8888:localhost:8888 your_username@login.hpc.ugent.be
# Visit http://localhost:8888 in browser
```

**Option 2: Job Submission (Recommended for Training)**

Create `submit_training.sh`:

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -t 04:00:00              # 4 hours
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --partition=gpu
#SBATCH -o training_%j.log       # Output log
#SBATCH --mail-type=END          # Email when done
#SBATCH --mail-user=your_email@ugent.be

cd ~/irisflow
source venv_ml/bin/activate

# Run Jupyter in batch mode (convert notebook to script)
jupyter nbconvert --to script notebooks/01_gaze_training.ipynb
python notebooks/01_gaze_training.py

# Or use papermill to run notebook with parameters
pip install papermill
papermill notebooks/01_gaze_training.ipynb notebooks/01_gaze_training_output.ipynb
```

Submit job:
```bash
sbatch submit_training.sh
```

Check status:
```bash
squeue -u your_username
```

---

## Week 2: Gaze Model Training

### Jupyter Notebook: `01_gaze_training.ipynb`

```python
# Cell 1: Load data
import json
import numpy as np
import pandas as pd
from pathlib import Path

with open("data/raw/gaze_dataset.json") as f:
    gaze_data = json.load(f)

# Convert to numpy arrays
landmarks = np.array([s['landmarks'] for s in gaze_data])
targets_x = np.array([s['screen_x'] for s in gaze_data])
targets_y = np.array([s['screen_y'] for s in gaze_data])

print(f"Loaded {len(gaze_data)} samples")
print(f"Landmarks shape: {landmarks.shape}")
print(f"Target X range: {targets_x.min()}-{targets_x.max()}")
print(f"Target Y range: {targets_y.min()}-{targets_y.max()}")

# Cell 2: Train/val/test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_x_train, y_x_test, y_y_train, y_y_test = \
    train_test_split(landmarks, targets_x, targets_y, 
                     test_size=0.30, random_state=42)

X_train, X_val, y_x_train, y_x_val, y_y_train, y_y_val = \
    train_test_split(X_train, y_x_train, y_y_train,
                     test_size=0.33, random_state=42)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Cell 3: Normalize landmarks
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Cell 4: PyTorch model
import torch
import torch.nn as nn
import torch.optim as optim

class GazeModel(nn.Module):
    def __init__(self, input_size=1404, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 2)  # Output: (x, y)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

model = GazeModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Model on device: {device}")

# Cell 5: Training loop
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100
patience = 10
best_val_loss = float('inf')
patience_counter = 0

train_losses = []
val_losses = []

for epoch in range(epochs):
    # Training
    model.train()
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_x_train_tensor = torch.FloatTensor(y_x_train).unsqueeze(1).to(device)
    y_y_train_tensor = torch.FloatTensor(y_y_train).unsqueeze(1).to(device)
    y_train_tensor = torch.cat([y_x_train_tensor, y_y_train_tensor], dim=1).to(device)
    
    pred_train = model(X_train_tensor)
    loss_train = criterion(pred_train, y_train_tensor)
    
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    
    train_losses.append(loss_train.item())
    
    # Validation
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        y_x_val_tensor = torch.FloatTensor(y_x_val).unsqueeze(1).to(device)
        y_y_val_tensor = torch.FloatTensor(y_y_val).unsqueeze(1).to(device)
        y_val_tensor = torch.cat([y_x_val_tensor, y_y_val_tensor], dim=1).to(device)
        
        pred_val = model(X_val_tensor)
        loss_val = criterion(pred_val, y_val_tensor)
    
    val_losses.append(loss_val.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train: {loss_train.item():.4f}, Val: {loss_val.item():.4f}")
    
    # Early stopping
    if loss_val < best_val_loss:
        best_val_loss = loss_val
        patience_counter = 0
        torch.save(model.state_dict(), "models/gaze_best.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Cell 6: Evaluate on test set
model.load_state_dict(torch.load("models/gaze_best.pt"))
model.eval()

with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_x_test_tensor = torch.FloatTensor(y_x_test).unsqueeze(1).to(device)
    y_y_test_tensor = torch.FloatTensor(y_y_test).unsqueeze(1).to(device)
    y_test_tensor = torch.cat([y_x_test_tensor, y_y_test_tensor], dim=1).to(device)
    
    pred_test = model(X_test_tensor).cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()

# Calculate RMSE
rmse = np.sqrt(np.mean((pred_test - y_test_np) ** 2, axis=0))
print(f"Test RMSE - X: {rmse[0]:.2f} px, Y: {rmse[1]:.2f} px")

# Cell 7: Export to ONNX
import torch.onnx

dummy_input = torch.randn(1, 1404).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "models/gaze.onnx",
    input_names=["landmarks"],
    output_names=["x", "y"],
    opset_version=11,
    do_constant_folding=True,
)
print("Exported to models/gaze.onnx")

# Save scaler for later normalization
import pickle
with open("models/gaze_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
```

**Execution on HPC:**

```bash
# Interactive session (for debugging)
jupyter notebook

# Or batch submission
sbatch submit_training.sh
```

**Monitoring:**
```bash
# Check job status
squeue -u your_username

# View logs in real-time
tail -f training_12345.log
```

---

## Week 3: Gesture LSTM Training

### Jupyter Notebook: `02_gesture_training.ipynb`

Similar structure:

```python
# Cell 1: Load gesture sequences
import pickle

with open("data/raw/gesture_sequences.pkl", "rb") as f:
    sequences = pickle.load(f)

# Prepare data
labels = [s['label'] for s in sequences]
landmarks_list = [s['landmarks'] for s in sequences]

label_to_idx = {label: i for i, label in enumerate(set(labels))}
y = np.array([label_to_idx[l] for l in labels])

# Convert to 3D array: [num_samples, 10, 630]
X = np.array(landmarks_list)
print(f"Data shape: {X.shape}, Target shape: {y.shape}")

# Cell 2: Train/val/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

# Cell 3: LSTM model
class GestureRecognitionLSTM(nn.Module):
    def __init__(self, input_size=630, hidden_size=128, num_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # [batch_size, hidden_size]
        logits = self.fc(last_hidden)
        return logits

model = GestureRecognitionLSTM()
model = model.to(device)

# Cell 4: Training with class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.FloatTensor(class_weights).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (similar to gaze, but for classification)
# ... (training code)

# Cell 5: Confusion matrix + per-class metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

with torch.no_grad():
    pred_test = model(X_test_tensor).argmax(dim=1).cpu().numpy()

acc = accuracy_score(y_test, pred_test)
print(f"Test Accuracy: {acc:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, pred_test, target_names=list(label_to_idx.keys())))

# Confusion matrix
cm = confusion_matrix(y_test, pred_test)
# Visualize...

# Cell 6: Export to ONNX
dummy_input = torch.randn(1, 10, 630).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "models/gesture.onnx",
    input_names=["sequences"],
    output_names=["logits"],
    opset_version=11,
)
```

---

## Download Models to M4

After training (Week 4), download ONNX models:

```bash
# From M4
scp your_username@login.hpc.ugent.be:/home/your_username/irisflow/models/gaze.onnx \
    models/

scp your_username@login.hpc.ugent.be:/home/your_username/irisflow/models/gesture.onnx \
    models/

# Or use rsync
rsync -avz --progress \
    your_username@login.hpc.ugent.be:/home/your_username/irisflow/models/ \
    models/
```

---

## Timeline with HPC

| Week | Task | Location | Duration |
|------|------|----------|----------|
| 1 | Data collection + validation | M4 | 7 days |
| 2 | Upload to HPC, gaze training | HPC (Jupyter) | 3-4 hours GPU |
| 3 | Gesture LSTM training | HPC (Jupyter) | 4-5 hours GPU |
| 4 | ONNX export, validation, download | HPC/M4 | 1-2 hours |
| 5-12 | Rust development + testing | M4 | 8 weeks |
| 13-16 | Polish, docs, release | M4 | 4 weeks |

**Key advantage:** While HPC trains models (4-5 hours total), you can work on Week 1 data validation or start reading `02_RUST_ARCHITECTURE.md`.

---

## Tips for HPC Training

1. **Batch size:** Start with 32-64; GPU memory is usually plentiful
2. **Checkpointing:** Save best model during training (done in notebook above)
3. **Logging:** Redirect stdout to file for later review
4. **Patience:** Jobs may queue; typical wait: 10 min - 2 hours
5. **Disk space:** Monitor with `df -h` and `quota` commands
6. **Time limits:** Start with 2-4 hour job limit; extend if needed

---

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size
- Use smaller model
- Run on CPU (slower but works)

### "Module not found (PyTorch, CUDA)"
```bash
# Ask HPC admins for correct modules
module avail PyTorch
module avail CUDA

# Or load specific versions
module load PyTorch/2.1.2-foss-2023b-CUDA-12.2.0
```

### "SSH connection drops during notebook"
- Use `tmux` to keep session alive
- Or submit job with papermill (don't need interactive connection)

### "Notebook kernel crashes"
- Increase memory limit: `#SBATCH --mem=32G`
- Reduce batch size in notebook

---

## Next Steps

1. **Verify HPC access:** Test login, check GPU availability
2. **Create directories:** Set up `/home/your_username/irisflow/` structure
3. **Upload code:** Push this repo to HPC (or use Git)
4. **Test setup:** Run simple Python script on GPU node
5. **Week 1:** Collect data locally (then follow timeline above)

---

## HPC vs. Local Comparison

| Aspect | M4 GPU (Metal) | RTX 3070 Ti | HPC GPU (A100/V100) |
|--------|---|---|---|
| **Inference** | Good (real-time) | Excellent | Overkill |
| **Training 5K samples** | 10 min (LSTM) | 5 min | 1 min |
| **Cost** | Free | Free | Free (included) |
| **Availability** | Always | Always | Queue time |
| **Best for** | Development | Final training | Hyperparameter sweep |

**Recommendation:** Use HPC for final training (Phase 1.3-1.4), M4 for iteration + Rust dev.
