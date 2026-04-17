# Data Collection Guide for IrisFlow ML Models

This document details how to collect high-quality training data for the gaze regression and gesture LSTM models.

---

## Part 1: Gaze Dataset Collection

### Overview
- **Goal:** 5000+ labeled gaze samples across the full screen
- **Ground truth:** Known screen position + eye landmarks from MediaPipe
- **Duration:** 30-60 minutes of recording
- **Environment:** Consistent lighting, normal head position, no glasses or sunglasses

### Methodology: 9-Point Grid with Repetition

1. **Display a calibration grid** on your M4 screen
   - 9 points (3×3) distributed across the display
   - Each point shown for 10 seconds
   - For each point, look at it steadily and record 5-10 samples at different head angles

2. **Head pose variation:**
   - Center (neutral): 3 samples
   - Left turn (±15°): 2 samples
   - Right turn (±15°): 2 samples
   - Down (±10°): 2 samples
   - Up (±10°): 2 samples
   - **Total per grid point:** ~11 samples

3. **Full collection breakdown:**
   - 9 grid points × 11 samples = ~99 samples per "round"
   - Run 50+ rounds = 5000+ samples
   - Estimated time: 50 min (99 samples / 2 samples per second)

### Implementation: Python Collection Script

Create `scripts/collect_gaze_data.py`:

```python
import cv2
import time
import json
import numpy as np
from pathlib import Path
from src.core.tracker import FaceTracker

# Configuration
GRID_POINTS = [
    (0.25, 0.25), (0.5, 0.25), (0.75, 0.25),  # Top row
    (0.25, 0.5),  (0.5, 0.5),  (0.75, 0.5),   # Middle row
    (0.25, 0.75), (0.5, 0.75), (0.75, 0.75),  # Bottom row
]
DISPLAY_SIZE = (1920, 1080)  # Adjust to your monitor
SAMPLE_DURATION_S = 0.5  # Seconds per sample
ROUNDS = 50

# Head pose variations: (yaw_deg, pitch_deg)
POSE_VARIATIONS = [
    (0, 0),      # Neutral
    (-15, 0),    # Left
    (15, 0),     # Right
    (0, -10),    # Down
    (0, 10),     # Up
]

def run_collection():
    tracker = FaceTracker()
    output_path = Path("data/gaze_dataset.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    samples = []
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting gaze data collection. Follow the red dots.")
    print(f"Grid points: {len(GRID_POINTS)}, Rounds: {ROUNDS}")
    print("Press 'q' to stop early, SPACE to skip a point.")
    
    for round_num in range(ROUNDS):
        for point_idx, (norm_x, norm_y) in enumerate(GRID_POINTS):
            for pose_idx, (yaw, pitch) in enumerate(POSE_VARIATIONS):
                # Display instruction
                print(f"\nRound {round_num+1}/{ROUNDS}, Point {point_idx+1}/9, "
                      f"Pose {pose_idx+1}/{len(POSE_VARIATIONS)}")
                print(f"  Position: ({norm_x:.0%}, {norm_y:.0%}), Yaw: {yaw}°, Pitch: {pitch}°")
                print("  Keep looking at the dot. Recording...")
                
                start_time = time.time()
                frame_count = 0
                
                while time.time() - start_time < SAMPLE_DURATION_S:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    # Get face landmarks
                    results = tracker.detect(frame)
                    if results and results.face_landmarks:
                        # Flatten landmarks: 468 points × 3 coords = 1404 dims
                        landmarks_flat = []
                        for lm in results.face_landmarks:
                            landmarks_flat.extend([lm.x, lm.y, lm.z])
                        
                        # Screen coordinates (absolute pixels)
                        screen_x = int(norm_x * DISPLAY_SIZE[0])
                        screen_y = int(norm_y * DISPLAY_SIZE[1])
                        
                        sample = {
                            "landmarks": landmarks_flat,
                            "screen_x": screen_x,
                            "screen_y": screen_y,
                            "norm_x": norm_x,
                            "norm_y": norm_y,
                            "yaw_deg": yaw,
                            "pitch_deg": pitch,
                            "round": round_num,
                            "grid_point": point_idx,
                            "pose_var": pose_idx,
                            "timestamp": time.time(),
                        }
                        samples.append(sample)
                        frame_count += 1
                    
                    # Display the dot on screen (in terminal or OpenCV window)
                    cv2.circle(frame, (int(norm_x * 640), int(norm_y * 480)), 10, (0, 0, 255), -1)
                    cv2.imshow("Gaze Collection", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Stopping early.")
                        break
                
                print(f"  Recorded {frame_count} frames.")
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save dataset
    with open(output_path, 'w') as f:
        json.dump(samples, f)
    
    print(f"\nCollection complete. Saved {len(samples)} samples to {output_path}")
    print(f"Expected: {ROUNDS * len(GRID_POINTS) * len(POSE_VARIATIONS)}")
    
    return samples

if __name__ == "__main__":
    run_collection()
```

### Quality Checks
- **Metadata verification:** Check that samples span full screen and multiple poses
- **Outlier detection:** Plot screen X vs Y; should cover full 2D space uniformly
- **Landmark quality:** Verify all 468 landmarks present (no NaNs)
- **Temporal coverage:** Samples spread across different times of day (vary lighting)

### Storage Format: `data/gaze_dataset.json`
```json
[
  {
    "landmarks": [x1, y1, z1, x2, y2, z2, ...],  // 468 landmarks × 3 = 1404 floats
    "screen_x": 480,                              // Pixel coordinate
    "screen_y": 270,
    "norm_x": 0.25,
    "norm_y": 0.25,
    "yaw_deg": 0,
    "pitch_deg": 0,
    "round": 5,
    "grid_point": 0,
    "pose_var": 0,
    "timestamp": 1713400000.123
  },
  ...
]
```

---

## Part 2: Gesture Dataset Collection

### Overview
- **Goal:** 2000+ labeled 10-frame gesture sequences
- **Gestures:** {open_hand, pinch, scroll_ready, swipe, palm, idle}
- **Duration:** 30-60 minutes of recording
- **Environment:** Good lighting, full hand visible in frame

### Methodology: Gesture-by-Gesture Recording

1. **Record each gesture type** in isolation
   - Open hand: Spread fingers apart, 30 sec video
   - Pinch: Thumb + index touching, 30 sec video
   - Scroll-ready: Two fingers extended (index + middle), 30 sec video
   - Swipe: Quick sideways movement, 30 sec video
   - Palm: All fingers extended, 30 sec video
   - Idle: Random hand movements, 60 sec video

2. **Extract 10-frame windows** from each recording
   - Sliding window: Every 5 frames, take 10-frame clip
   - Each 30-sec video @ 30 FPS = 900 frames = ~180 windows
   - Each gesture type: 6 × 180 = 1080 windows (with 6 videos per gesture)

3. **Augmentation:**
   - Hand scale variation: Record with hand far/near camera
   - Hand angle: Rotate hand in frame
   - Lighting: Different backgrounds, lighting angles

### Implementation: Gesture Collection Script

Create `scripts/collect_gesture_data.py`:

```python
import cv2
import time
import pickle
import numpy as np
from pathlib import Path
from src.core.tracker import HandTracker

GESTURES = ['open_hand', 'pinch', 'scroll_ready', 'swipe', 'palm', 'idle']
RECORDING_DURATION_S = 30  # Per gesture (except idle: 60s)
WINDOW_SIZE = 10  # Frames per training sample

def run_collection():
    tracker = HandTracker()
    output_path = Path("data/gesture_sequences.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    sequences = []
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Starting gesture data collection.")
    print(f"Gestures: {GESTURES}")
    print("Press SPACE to start recording, 'q' to stop.\n")
    
    for gesture in GESTURES:
        duration = 60 if gesture == 'idle' else RECORDING_DURATION_S
        print(f"\nGesture: {gesture}")
        print(f"Perform the gesture for {duration} seconds. Press SPACE when ready.")
        
        # Wait for user to press space
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.putText(frame, f"Press SPACE to start: {gesture}", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Gesture Collection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
        # Record gesture
        print(f"Recording {gesture}... (Press 'q' to stop early)")
        start_time = time.time()
        gesture_frames = []
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Detect hand landmarks
            results = tracker.detect(frame)
            
            if results and results.hand_landmarks:
                # Store landmarks for both hands (if present)
                for hand_landmarks in results.hand_landmarks:
                    landmarks_flat = []
                    for lm in hand_landmarks:
                        landmarks_flat.extend([lm.x, lm.y, lm.z])
                    gesture_frames.append(landmarks_flat)
            
            # Display progress
            elapsed = time.time() - start_time
            cv2.putText(frame, f"{gesture} {elapsed:.1f}s/{duration}s", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Gesture Collection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"Stopped early. Got {len(gesture_frames)} frames.")
                break
        
        # Extract 10-frame windows
        for i in range(0, len(gesture_frames) - WINDOW_SIZE, 5):  # Stride of 5
            window = gesture_frames[i:i+WINDOW_SIZE]
            if len(window) == WINDOW_SIZE:
                sequence = {
                    "landmarks": window,  # List of 10 frames, each 63 dims (21×3)
                    "label": gesture,
                    "source_frame": i,
                }
                sequences.append(sequence)
        
        print(f"Extracted {len([s for s in sequences if s['label'] == gesture])} windows for {gesture}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save dataset
    with open(output_path, 'wb') as f:
        pickle.dump(sequences, f)
    
    print(f"\nCollection complete. Saved {len(sequences)} sequences to {output_path}")
    
    # Print distribution
    label_counts = {}
    for seq in sequences:
        label = seq['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("Label distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    return sequences

if __name__ == "__main__":
    run_collection()
```

### Storage Format: `data/gesture_sequences.pkl`
```python
[
  {
    "landmarks": [
      [x1, y1, z1, ..., x21, y21, z21],  # Frame 1 (21 hand landmarks × 3)
      [x1, y1, z1, ..., x21, y21, z21],  # Frame 2
      ...
      [x1, y1, z1, ..., x21, y21, z21],  # Frame 10
    ],
    "label": "pinch",
    "source_frame": 45,
  },
  ...
]
```

---

## Part 3: Data Validation & Cleaning

### Automated Checks

```python
# scripts/validate_datasets.py

def validate_gaze_dataset(gaze_data):
    """Check for data quality issues."""
    issues = []
    
    # Check coverage
    screen_xs = [s['screen_x'] for s in gaze_data]
    screen_ys = [s['screen_y'] for s in gaze_data]
    
    if max(screen_xs) - min(screen_xs) < 1500:  # Should span most of width
        issues.append("WARNING: Limited X-axis coverage")
    if max(screen_ys) - min(screen_ys) < 800:   # Should span most of height
        issues.append("WARNING: Limited Y-axis coverage")
    
    # Check for NaNs/Infs
    for i, sample in enumerate(gaze_data):
        landmarks = sample['landmarks']
        if any(np.isnan(landmarks)) or any(np.isinf(landmarks)):
            issues.append(f"Sample {i}: Contains NaN or Inf in landmarks")
    
    # Check timestamp ordering
    for i in range(1, len(gaze_data)):
        if gaze_data[i]['timestamp'] < gaze_data[i-1]['timestamp']:
            issues.append(f"Sample {i}: Out of order timestamp")
    
    return issues

def validate_gesture_dataset(gesture_data):
    """Check gesture sequences."""
    issues = []
    
    # Check label distribution
    labels = {}
    for seq in gesture_data:
        label = seq['label']
        labels[label] = labels.get(label, 0) + 1
    
    for label, count in labels.items():
        if count < 100:
            issues.append(f"WARNING: {label} has only {count} sequences (target: >100)")
    
    # Check sequence length
    for i, seq in enumerate(gesture_data):
        if len(seq['landmarks']) != 10:
            issues.append(f"Sequence {i}: Expected 10 frames, got {len(seq['landmarks'])}")
    
    return issues
```

### Visualization

```python
# scripts/visualize_datasets.py

import matplotlib.pyplot as plt

def plot_gaze_distribution(gaze_data):
    """Scatter plot of gaze samples on screen."""
    screen_xs = [s['screen_x'] for s in gaze_data]
    screen_ys = [s['screen_y'] for s in gaze_data]
    
    plt.scatter(screen_xs, screen_ys, alpha=0.5, s=10)
    plt.xlabel("Screen X")
    plt.ylabel("Screen Y")
    plt.title("Gaze Sample Distribution")
    plt.grid()
    plt.show()

def plot_gesture_distribution(gesture_data):
    """Bar chart of gesture label counts."""
    labels = {}
    for seq in gesture_data:
        label = seq['label']
        labels[label] = labels.get(label, 0) + 1
    
    plt.bar(labels.keys(), labels.values())
    plt.xlabel("Gesture")
    plt.ylabel("Count")
    plt.title("Gesture Sample Distribution")
    plt.xticks(rotation=45)
    plt.show()
```

---

## Tips for High-Quality Data

1. **Lighting:** Record in consistent lighting. Avoid backlit scenes (window behind you).
2. **Camera angle:** Position camera slightly below eye level, facing you directly.
3. **Rest and breaks:** Take 5-minute breaks every 15 minutes to avoid fatigue.
4. **Natural movement:** Don't be too stiff. Move your head naturally during gaze collection.
5. **Multiple sessions:** If possible, collect data across different days/times for robustness.
6. **Verify landmarks:** Periodically check that MediaPipe is detecting faces/hands reliably.

---

## Next Steps

1. Create collection scripts (this week)
2. Run gaze collection (Day 1-2)
3. Validate gaze data quality, visualize (Day 3)
4. Run gesture collection (Day 4-5)
5. Validate gesture data, check label balance (Day 6)
6. Ready for model training (Week 2)
