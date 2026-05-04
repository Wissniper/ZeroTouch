# ML Training & HPC Workflow

## 1. The "Golden Path" Architecture
To achieve high performance, we use a hybrid workflow:
1. **Research & Training**: Python (PyTorch/TensorFlow)
2. **Export**: ONNX (Open Neural Network Exchange)
3. **Production Inference**: C++ (ONNX Runtime)

## 2. High-Performance Computing (HPC) Setup
Training deep learning models for iris tracking requires significant GPU power (e.g., NVIDIA A100 or H100 clusters).

### Prerequisites:
- **Environment**: Anaconda/Miniconda
- **Compute**: SLURM Workload Manager (common in university/research clusters)
- **Data**: Large-scale iris datasets (e.g., MPIIGaze, Gaze360)

### Training Workflow:
1. **Pre-processing**: Normalize images to $224 \times 224$ pixels.
2. **Model Architecture**: Use a lightweight backbone like **MobileNetV3** or **EfficientNet-Lite** (necessary for 60 FPS).
3. **Loss Functions**: 
   - **Gaze**: Euclidean distance loss on the gaze vector.
   - **Gestures**: Cross-entropy loss for classification.

## 3. HPC Submission Script (Example)
Use this `sbatch` script to submit training jobs to an HPC cluster:

```bash
#!/bin/bash
#SBATCH --job-name=irisflow_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00

module load cuda/11.8
source activate irisflow_env

# Start training
python train_gaze_model.py --epochs 100 --batch_size 128 --model mobilenet_v3
```

## 4. Exporting to C++ (ONNX)
Once training is complete in Python, export the model to the ONNX format so the C++ binary can load it:

```python
import torch

# Load your trained model
model = MyGazeModel()
model.load_state_dict(torch.load("gaze_weights.pt"))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export
torch.onnx.export(model, dummy_input, "gaze.onnx", 
                  input_names=['input'], 
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}})
```

## 5. Integration
Place the resulting `.onnx` file in the `irisflow-cpp/models/` folder. The `ONNXEngine` in the C++ project will automatically load it for real-time control.
