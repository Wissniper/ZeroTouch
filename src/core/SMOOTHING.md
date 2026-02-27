# IrisFlow Smoothing Roadmap: Signal Stabilization Strategy

## 1. The Problem: Sensor Noise vs. Human Physiology
Iris tracking suffers from two distinct types of "jitter":
1. **Sensor Noise:** MediaPipe's landmark estimation fluctuates slightly between frames due to lighting and camera resolution.
2. **Micro-saccades:** The human eye does not stay perfectly still; it makes constant, tiny involuntary jumps.

Without smoothing, the cursor will vibrate, making it impossible to click small UI elements.

---

## 2. Proposed Algorithms (Low to High Complexity)

### A. Exponential Moving Average (EMA)
**Concept:** Every new coordinate is a weighted average of the current raw input and the previous smoothed output.
**Formula:** `y_t = α * x_t + (1 - α) * y_{t-1}`
*   **α (Alpha):** The "Smoothing Factor" (0.0 to 1.0).
*   **Pros:** Extremely lightweight; easy to implement.
*   **Cons:** Introduces "lag" if smoothed too much. If α is low, the cursor feels "heavy."

### B. One-Euro Filter (Recommended)
**Concept:** An adaptive low-pass filter that changes its smoothing strength based on velocity.
*   **At Low Speeds:** High smoothing to eliminate jitter while the user is trying to aim at a button.
*   **At High Speeds:** Low smoothing to reduce lag while the user is moving the cursor across the screen.
*   **Parameters:** `min_cutoff`, `beta`, `d_cutoff`.
*   **Pros:** Industry standard for human-computer interaction; minimizes the trade-off between jitter and lag.

### C. Moving Median Filter
**Concept:** Takes the median of the last *N* frames (e.g., last 5 frames).
*   **Pros:** Excellent at removing "spikes" or "outliers" (e.g., if the tracker loses the eye for a single frame).
*   **Cons:** Higher latency than EMA.

---

## 3. Implementation Blueprint

### Step 1: The Signal Buffer
Smoothing requires **state**. You cannot smooth a single frame in isolation.
*   A new class (e.g., `SmoothingFilter`) must be created to store the coordinates of the previous frame.
*   This should be placed in `src/core/processor.py`.

### Step 2: The Integration Pipeline
The smoothing must occur at a specific point in the data flow:
1. **Raw Landmarks** (from `tracker.py`)
2. **Normalized Gaze Ratio** (from `tracker.py`)
3. **Screen Mapping** (from `mapper.py`)
4. **SMOOTHING** (The new step)
5. **Cursor Action** (via `pyautogui`)

### Step 3: Parameter Tuning (The "Feel")
Smoothing is subjective. The `SMOOTHING.md` recommends a "Tuning Phase" where:
*   **Static Jitter:** Watch the cursor while staring at one spot. Increase smoothing until the vibration stops.
*   **Dynamic Lag:** Move your eyes quickly. Decrease smoothing if the cursor feels like it's "chasing" your gaze too slowly.

---

## 4. Visual Debugging Strategy
To verify smoothing without moving the mouse, the following should be rendered on the OpenCV feed:
1.  **Red Dot:** Raw, jittery iris position.
2.  **Green Dot:** The stabilized, smoothed position.
*When the green dot follows the red dot smoothly without vibrating, the algorithm is successful.*
