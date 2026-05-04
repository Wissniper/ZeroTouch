# Kalman Filter for Gaze Stabilization

## 1. The Problem: Jitter and Noise
In eye tracking, the raw data from the webcam is inherently noisy. Lighting changes, sensor noise, and micro-movements of the eye (microsaccades) cause the cursor to "shake" or jitter even when you are looking at a fixed point. A simple average of frames (Moving Average) introduces too much lag—the cursor feels like it's dragging through honey.

## 2. What is a Kalman Filter?
The Kalman Filter is an optimal estimation algorithm. It doesn't just "smooth" data; it **predicts** where the eye is going based on its velocity and then **corrects** that prediction using the actual measurement from the camera.

### The Two-Step Cycle:
1.  **Predict**: "Based on the last position and velocity, where should the eye be right now?"
2.  **Update (Correct)**: "The camera says the eye is *here*. How much should I trust the camera versus my prediction?"

## 3. Mathematical Intuition
We model the eye as a point with a position $(x, y)$ and a velocity $(v_x, v_y)$.

### The State Vector:
$$ \hat{x} = \begin{bmatrix} x \\ y \\ v_x \\ v_y \end{bmatrix} $$

### The Transition Matrix ($A$):
This matrix tells the filter how the state evolves over time ($dt$):
$$ x_{new} = x_{old} + v_x \cdot dt $$
$$ y_{new} = y_{old} + v_y \cdot dt $$

### The Kalman Gain ($K$):
This is the "secret sauce." If the camera measurement is very noisy (low confidence), $K$ becomes small, and the filter trusts its internal prediction more. If the camera is very accurate, $K$ becomes large, and the filter trusts the measurement more.

## 4. Implementation in IrisFlow
In `src/processing/KalmanFilter.cpp`, we use a 4-state Kalman Filter.
-   **High Accuracy**: It allows the cursor to be perfectly still when your eye is still.
-   **Low Lag**: Because it understands velocity, the cursor "snaps" to new locations almost instantly when you move your eyes quickly.
