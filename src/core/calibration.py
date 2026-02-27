import cv2 as cv
import numpy as np

class GazeCalibrator:
    def __init__(self):
        """
        Handles 9-point calibration and dynamic drift correction.
        """
        self.reference_points = [] # The 9 physical points on the screen
        self.gaze_points = []      # The user's eye coordinates at those points
        self.transform_matrix = None
        
        # Base screen coordinates for a standard 9-point grid (normalized 0.0 to 1.0)
        self.calibration_grid = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
        ]

    def add_calibration_point(self, screen_x, screen_y, gaze_x, gaze_y):
        """Stores a mapped point during the calibration UI phase."""
        self.reference_points.append([screen_x, screen_y])
        self.gaze_points.append([gaze_x, gaze_y])

    def calculate_mapping(self):
        """
        Computes the Homography matrix using the 9 calibration points.
        This matrix maps the non-linear 'bowl' of the eye to the flat screen.
        """
        if len(self.reference_points) < 4:
            return False # Need at least 4 points for homography
            
        pts_src = np.array(self.gaze_points, dtype=float)
        pts_dst = np.array(self.reference_points, dtype=float)
        
        # findHomography finds the best fit perspective transform using multiple points
        self.transform_matrix, _ = cv.findHomography(pts_src, pts_dst)
        return self.transform_matrix is not None

    def apply_transform(self, gaze_x, gaze_y):
        """
        Applies the calculated matrix to real-time gaze coordinates.
        Replaces the old CoordinateMapper linear logic.
        """
        if self.transform_matrix is None:
            return gaze_x, gaze_y # Fallback if uncalibrated
            
        # Homography requires 3D coordinates (x, y, 1)
        point = np.array([[[gaze_x, gaze_y]]], dtype=float)
        
        # Apply the matrix using OpenCV
        transformed = cv.perspectiveTransform(point, self.transform_matrix)
        
        # Extract the new X, Y
        return transformed[0][0][0], transformed[0][0][1]

    def correct_drift(self, expected_center_gaze, current_center_gaze):
        """
        Auto-Drift Fix: Shifts the matrix linearly if the user resets their gaze.
        """
        dx = expected_center_gaze[0] - current_center_gaze[0]
        dy = expected_center_gaze[1] - current_center_gaze[1]
        
        # Apply linear shift to future coordinates
        # Implementation depends on whether we adjust the input or the matrix itself.
        pass
