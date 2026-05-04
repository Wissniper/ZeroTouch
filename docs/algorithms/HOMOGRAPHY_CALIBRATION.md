# Homography & Calibration Explained

## 1. The Problem: Perspective Distortion
Your eye is a sphere sitting in a socket. When you look at a screen, the relationship between where your iris is located in the webcam image and where you are looking on the screen is not a simple linear map. It is a **projective transformation**. 

If you look at the top-left of the screen versus the center, the iris moves in a way that looks like it's being "projected" onto a 2D plane from a curved surface.

## 2. What is a Homography?
In computer vision, a **Homography** is a matrix that maps points from one plane (the eye-socket plane) to another plane (the computer screen). 

It is a $3 \times 3$ matrix that can handle:
- **Translation**: Moving left/right/up/down.
- **Scaling**: Zooming in/out.
- **Rotation**: Turning.
- **Shearing/Perspective**: Correcting for the angle of the camera relative to your face.

## 3. The Calibration Process (The 9 Dots)
To calculate this matrix, we need corresponding points between the two planes. 
- We show you a dot at a known screen position (e.g., $0,0$).
- We record where your iris is in the camera image when you look at that dot (e.g., $0.34, 0.51$).
- After 9 points, we have enough data to solve for the 8 variables in the Homography matrix.

## 4. RANSAC: Dealing with Human Error
Humans blink, get distracted, or look away during calibration. These are "outliers."
**RANSAC (Random Sample Consensus)** is an algorithm used during the homography calculation:
1. It picks 4 random points from your 9 calibration points.
2. It calculates a "candidate" homography.
3. It checks how many of the *other* points fit this candidate.
4. It repeats this thousands of times and picks the version that fits the most points.
5. This ensures that if you messed up one or two dots, the calibration still works perfectly.

## 5. Implementation in IrisFlow
In `src/processing/GazeCalibrator.cpp`, we use `cv::findHomography` with the `cv::RANSAC` flag. This makes the eye-to-screen mapping robust against accidental blinks or minor head shifts during setup.
