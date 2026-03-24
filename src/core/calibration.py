"""9-point gaze calibration via homography with drift correction.

The homography maps the non-linear "bowl" of iris-in-eye-socket ratios
to flat screen pixel coordinates.  RANSAC is used during findHomography
to reject outlier calibration points.
"""

import cv2 as cv
import numpy as np


class GazeCalibrator:
    """Collects calibration samples, computes a homography, and applies it."""

    # Standard 9-point grid in normalized [0,1] coords
    CALIBRATION_GRID = [
        (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
        (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
        (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
    ]

    def __init__(self):
        self.reference_points: list[tuple[float, float]] = []
        self.gaze_points: list[tuple[float, float]] = []
        self.transform_matrix: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Calibration collection
    # ------------------------------------------------------------------

    def get_current_point(self, screen_w: int, screen_h: int, index: int):
        """Pixel coordinates for calibration dot *index* (0-8)."""
        rel_x, rel_y = self.CALIBRATION_GRID[index]
        return int(rel_x * screen_w), int(rel_y * screen_h)

    def add_calibration_point(self, screen_x, screen_y, gaze_x, gaze_y):
        """Store a (screen target) ↔ (gaze reading) pair."""
        self.reference_points.append((screen_x, screen_y))
        self.gaze_points.append((gaze_x, gaze_y))

    def is_finished(self) -> bool:
        """True when all 9 calibration samples have been collected."""
        return len(self.gaze_points) >= 9

    # ------------------------------------------------------------------
    # Homography computation
    # ------------------------------------------------------------------

    def calculate_mapping(self) -> bool:
        """Compute the homography matrix from collected calibration pairs.

        Uses RANSAC (cv.findHomography default method=0 is least-squares;
        we explicitly pass RANSAC for outlier robustness with
        reprojectionThreshold=5.0 pixels).

        Returns True if the matrix was computed successfully.
        """
        if len(self.reference_points) < 4:
            return False

        pts_src = np.array(self.gaze_points, dtype=np.float64)
        pts_dst = np.array(self.reference_points, dtype=np.float64)

        self.transform_matrix, mask = cv.findHomography(
            pts_src, pts_dst, cv.RANSAC, 5.0
        )

        if self.transform_matrix is not None and mask is not None:
            inliers = int(mask.sum())
            total = len(mask)
            if inliers < 4:
                # Too few inliers — calibration data is unreliable
                self.transform_matrix = None
                return False

        return self.transform_matrix is not None

    def apply_transform(self, gaze_x: float, gaze_y: float) -> tuple[float, float]:
        """Map a gaze ratio pair to screen pixels via the homography.

        Falls back to identity (returns input unchanged) if uncalibrated.
        """
        if self.transform_matrix is None:
            return gaze_x, gaze_y

        point = np.array([[[gaze_x, gaze_y]]], dtype=np.float64)
        transformed = cv.perspectiveTransform(point, self.transform_matrix)
        return float(transformed[0][0][0]), float(transformed[0][0][1])

    # ------------------------------------------------------------------
    # Drift correction
    # ------------------------------------------------------------------

    def correct_drift(
        self,
        expected_screen: tuple[float, float],
        current_gaze: tuple[float, float],
    ) -> None:
        """Apply a translational offset to the homography to correct drift.

        When the user re-centres their gaze (e.g. looks at screen centre),
        the *expected* screen position is known. If the homography maps the
        current gaze to a different spot, the difference is a drift vector.

        We bake this offset directly into the homography's translation
        column (H[0,2] and H[1,2]) so all future apply_transform calls
        are automatically corrected.

        Args:
            expected_screen: Where the cursor *should* be (pixels).
            current_gaze: The raw gaze ratio the user is producing right now.
        """
        if self.transform_matrix is None:
            return

        # Where the homography currently maps the gaze
        actual_x, actual_y = self.apply_transform(*current_gaze)

        # Drift = expected minus actual
        dx = expected_screen[0] - actual_x
        dy = expected_screen[1] - actual_y

        # Shift the translation component of the 3×3 homography
        self.transform_matrix[0, 2] += dx
        self.transform_matrix[1, 2] += dy
