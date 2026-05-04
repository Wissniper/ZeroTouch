# Gesture Stability & Performance Optimizations

## 1. Temporal Filtering (Majority Voting)
### The Problem: Frame Flickering
In the Python prototype, gestures were detected one frame at a time. If the camera had a bit of motion blur for just 1/30th of a second, the "palm" gesture might look like a "fist" for a single frame. This caused the cursor to "flicker" between actions, like rapidly clicking or scrolling.

### The Solution: 5-Frame Majority Vote
Instead of acting on every frame, the C++ implementation keeps a **History Queue** of the last 5 frames.
- Frame 1: SCROLL
- Frame 2: SCROLL
- Frame 3: NONE (noise)
- Frame 4: SCROLL
- Frame 5: SCROLL
**The Result**: The system counts the occurrences. SCROLL (4) > NONE (1). The system executes SCROLL. 
This "debouncing" logic ensures that the system only reacts to intentional, sustained gestures.

## 2. ROI Tracking (Region of Interest)
### The Problem: Full Frame Waste
A typical webcam frame is $640 \times 480$ pixels. Your face might only take up $200 \times 200$ pixels in the center. Running a heavy Neural Network on all those empty background pixels is a waste of CPU/GPU power and slows down your FPS.

### The Solution: Crop and Track
1. **Initial Search**: We look at the full frame once to find your face.
2. **Crop (ROI)**: We define a small box around your eyes and hands.
3. **Focused Inference**: In the next frame, we only look inside that small box. 
4. **Update**: We move the box as your face moves.
This optimization is why the C++ version can reach 60+ FPS while the Python version struggled at 20-30 FPS.

## 3. Confidence Gating
### The Problem: Hallucinations
When you move your hand out of the camera's view, the Neural Network will often "hallucinate" landmarks in the background noise, trying to find a hand where there isn't one. This leads to "ghost clicks."

### The Solution: The 0.7 Threshold
Every prediction from an ML model comes with a **Confidence Score** (0 to 1). 
- If score is $0.9$: "I am 90% sure this is an eye."
- If score is $0.4$: "I think this is an eye, but I'm guessing."
The C++ implementation uses a **Hard Gate**: if the score is below $0.7$, the data is discarded instantly. No move or click is executed. This makes the system feel much more professional and "quiet" when you aren't using it.
