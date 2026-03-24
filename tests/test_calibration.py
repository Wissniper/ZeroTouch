"""Tests for GazeCalibrator — homography computation and drift correction.

Tests use synthetic calibration data where we know the exact mapping,
so we can verify the homography reconstructs it accurately.
"""

import numpy as np
import pytest
from src.core.calibration import GazeCalibrator


class TestGazeCalibrator:
    def _make_calibrator_with_identity(self):
        """Create a calibrator with a simple linear mapping for testing.

        Gaze points are in [0,1] normalized range.
        Screen points are gaze * 1920 or * 1080 (simple scale).
        """
        cal = GazeCalibrator()
        screen_w, screen_h = 1920, 1080
        for i in range(9):
            sx, sy = cal.get_current_point(screen_w, screen_h, i)
            # Simulate a perfectly linear gaze→screen relationship
            gx = sx / screen_w
            gy = sy / screen_h
            cal.add_calibration_point(sx, sy, gx, gy)
        return cal, screen_w, screen_h

    def test_get_current_point_range(self):
        """All 9 calibration points should be within screen bounds."""
        cal = GazeCalibrator()
        for i in range(9):
            x, y = cal.get_current_point(1920, 1080, i)
            assert 0 <= x <= 1920
            assert 0 <= y <= 1080

    def test_add_calibration_point(self):
        cal = GazeCalibrator()
        cal.add_calibration_point(100, 200, 0.5, 0.5)
        assert len(cal.reference_points) == 1
        assert len(cal.gaze_points) == 1

    def test_is_finished(self):
        cal = GazeCalibrator()
        assert not cal.is_finished()
        for i in range(9):
            cal.add_calibration_point(i, i, i * 0.1, i * 0.1)
        assert cal.is_finished()

    def test_calculate_mapping_needs_min_4_points(self):
        cal = GazeCalibrator()
        for i in range(3):
            cal.add_calibration_point(i * 100, i * 100, i * 0.1, i * 0.1)
        assert cal.calculate_mapping() is False

    def test_calculate_mapping_succeeds_with_9_points(self):
        cal, _, _ = self._make_calibrator_with_identity()
        assert cal.calculate_mapping() is True
        assert cal.transform_matrix is not None
        assert cal.transform_matrix.shape == (3, 3)

    def test_apply_transform_reconstructs_mapping(self):
        """The homography should accurately map gaze→screen for known points."""
        cal, screen_w, screen_h = self._make_calibrator_with_identity()
        cal.calculate_mapping()

        # Test the centre point
        sx, sy = cal.apply_transform(0.5, 0.5)
        assert abs(sx - screen_w * 0.5) < 10, f"Centre X off: {sx}"
        assert abs(sy - screen_h * 0.5) < 10, f"Centre Y off: {sy}"

    def test_apply_transform_corners(self):
        """Verify corner mapping accuracy."""
        cal, screen_w, screen_h = self._make_calibrator_with_identity()
        cal.calculate_mapping()

        corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
        for gx, gy in corners:
            sx, sy = cal.apply_transform(gx, gy)
            expected_x = gx * screen_w
            expected_y = gy * screen_h
            assert abs(sx - expected_x) < 20, f"Corner ({gx},{gy}) X: {sx} vs {expected_x}"
            assert abs(sy - expected_y) < 20, f"Corner ({gx},{gy}) Y: {sy} vs {expected_y}"

    def test_apply_transform_uncalibrated_passthrough(self):
        """Without calibration, apply_transform should return input unchanged."""
        cal = GazeCalibrator()
        x, y = cal.apply_transform(0.5, 0.5)
        assert x == 0.5
        assert y == 0.5

    def test_correct_drift(self):
        """Drift correction should shift future transforms."""
        cal, screen_w, screen_h = self._make_calibrator_with_identity()
        cal.calculate_mapping()

        # Before drift: centre maps to ~(960, 540)
        before_x, before_y = cal.apply_transform(0.5, 0.5)

        # Apply a drift of 50px right, 30px down
        cal.correct_drift(
            expected_screen=(before_x + 50, before_y + 30),
            current_gaze=(0.5, 0.5),
        )

        after_x, after_y = cal.apply_transform(0.5, 0.5)
        assert abs(after_x - (before_x + 50)) < 1.0
        assert abs(after_y - (before_y + 30)) < 1.0

    def test_correct_drift_no_op_when_uncalibrated(self):
        """Drift correction on uncalibrated instance should not crash."""
        cal = GazeCalibrator()
        cal.correct_drift((960, 540), (0.5, 0.5))  # should be a no-op

    def test_grid_has_9_points(self):
        assert len(GazeCalibrator.CALIBRATION_GRID) == 9

    def test_grid_symmetric(self):
        """Grid should be symmetric around 0.5."""
        grid = GazeCalibrator.CALIBRATION_GRID
        xs = sorted(set(g[0] for g in grid))
        ys = sorted(set(g[1] for g in grid))
        assert xs == [0.1, 0.5, 0.9]
        assert ys == [0.1, 0.5, 0.9]
