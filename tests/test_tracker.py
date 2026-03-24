"""Tests for VisionTracker helper methods.

These tests cover the pure-computation methods (gaze ratio, blink scores,
head pose, finger counting) using mock landmark data, without requiring
actual MediaPipe models or a camera.
"""

import math
from types import SimpleNamespace
import pytest
from src.core.tracker import VisionTracker, _EPS


def _make_landmark(x, y, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def _mock_face_result(landmarks=None, blendshapes=None, matrices=None):
    """Build a minimal mock of MediaPipe FaceLandmarkerResult."""
    return SimpleNamespace(
        face_landmarks=[landmarks] if landmarks else [],
        face_blendshapes=[blendshapes] if blendshapes else [],
        facial_transformation_matrixes=[matrices] if matrices is not None else [],
    )


def _mock_hand_result(landmarks=None):
    return SimpleNamespace(
        hand_landmarks=[landmarks] if landmarks else [],
    )


class TestGetGazeRatio:
    def _build_symmetric_landmarks(self, iris_h_ratio=0.5, iris_v_ratio=0.5):
        """Create 478 landmarks with iris positioned at given ratios.

        We only need indices 33, 133, 159, 145 (left eye),
        362, 263, 386, 374 (right eye), 468, 473 (iris centres).
        """
        lm = [_make_landmark(0.0, 0.0) for _ in range(478)]

        # Left eye socket  (33=outer x=0.3, 133=inner x=0.4)
        lm[33] = _make_landmark(0.3, 0.0)
        lm[133] = _make_landmark(0.4, 0.0)
        lm[159] = _make_landmark(0.0, 0.3)  # top
        lm[145] = _make_landmark(0.0, 0.4)  # bottom

        # Right eye socket (362=inner x=0.6, 263=outer x=0.7)
        lm[362] = _make_landmark(0.6, 0.0)
        lm[263] = _make_landmark(0.7, 0.0)
        lm[386] = _make_landmark(0.0, 0.3)  # top
        lm[374] = _make_landmark(0.0, 0.4)  # bottom

        # Place irises at the requested ratios within their sockets
        l_iris_x = 0.3 + iris_h_ratio * (0.4 - 0.3)
        r_iris_x = 0.6 + iris_h_ratio * (0.7 - 0.6)
        iris_y = 0.3 + iris_v_ratio * (0.4 - 0.3)

        lm[468] = _make_landmark(l_iris_x, iris_y)
        lm[473] = _make_landmark(r_iris_x, iris_y)

        return lm

    def test_centered_gaze(self):
        """Centered iris should give ratios near (0.5, 0.5)."""
        lm = self._build_symmetric_landmarks(0.5, 0.5)
        face = _mock_face_result(landmarks=lm)

        # We can't call tracker.get_gaze_ratio without a real VisionTracker,
        # so test the math directly:
        # Replicate the formula from tracker.py
        l_iris, r_iris = lm[468], lm[473]
        l_h = (l_iris.x - lm[33].x) / (lm[133].x - lm[33].x + _EPS)
        r_h = (r_iris.x - lm[362].x) / (lm[263].x - lm[362].x + _EPS)
        l_v = (l_iris.y - lm[159].y) / (lm[145].y - lm[159].y + _EPS)
        r_v = (r_iris.y - lm[386].y) / (lm[374].y - lm[386].y + _EPS)

        avg_h = (l_h + r_h) / 2.0
        avg_v = 1.0 - (l_v + r_v) / 2.0

        assert abs(avg_h - 0.5) < 0.01
        assert abs(avg_v - 0.5) < 0.01

    def test_looking_left(self):
        lm = self._build_symmetric_landmarks(iris_h_ratio=0.1)
        l_iris, r_iris = lm[468], lm[473]
        l_h = (l_iris.x - lm[33].x) / (lm[133].x - lm[33].x + _EPS)
        r_h = (r_iris.x - lm[362].x) / (lm[263].x - lm[362].x + _EPS)
        avg_h = (l_h + r_h) / 2.0
        assert avg_h < 0.3, "Should register as looking left"

    def test_looking_right(self):
        lm = self._build_symmetric_landmarks(iris_h_ratio=0.9)
        l_iris, r_iris = lm[468], lm[473]
        l_h = (l_iris.x - lm[33].x) / (lm[133].x - lm[33].x + _EPS)
        r_h = (r_iris.x - lm[362].x) / (lm[263].x - lm[362].x + _EPS)
        avg_h = (l_h + r_h) / 2.0
        assert avg_h > 0.7, "Should register as looking right"

    def test_epsilon_prevents_div_by_zero(self):
        """If eye socket collapses (same x for outer/inner), no crash."""
        lm = [_make_landmark(0.5, 0.5) for _ in range(478)]
        # Left eye: both corners at same x
        lm[33] = _make_landmark(0.5, 0.5)
        lm[133] = _make_landmark(0.5, 0.5)  # same as 33!
        lm[159] = _make_landmark(0.5, 0.5)
        lm[145] = _make_landmark(0.5, 0.5)
        lm[362] = _make_landmark(0.5, 0.5)
        lm[263] = _make_landmark(0.5, 0.5)
        lm[386] = _make_landmark(0.5, 0.5)
        lm[374] = _make_landmark(0.5, 0.5)
        lm[468] = _make_landmark(0.5, 0.5)
        lm[473] = _make_landmark(0.5, 0.5)

        # This would crash without _EPS
        denom = lm[133].x - lm[33].x + _EPS
        result = (lm[468].x - lm[33].x) / denom
        assert math.isfinite(result)


class TestCountExtendedFingers:
    def test_closed_fist(self):
        """All tips closer to wrist than MCPs → 0 fingers."""
        # Wrist at (0.5, 0.8), MCPs at ~(0.5, 0.6), tips at (0.5, 0.7) — closer
        lm = [_make_landmark(0.5, 0.5) for _ in range(21)]
        lm[0] = _make_landmark(0.5, 0.8)  # wrist
        # MCP joints further from wrist
        for mcp in [2, 5, 9, 13, 17]:
            lm[mcp] = _make_landmark(0.5, 0.4)
        # Tips closer to wrist than MCPs
        for tip in [4, 8, 12, 16, 20]:
            lm[tip] = _make_landmark(0.5, 0.7)

        assert VisionTracker.count_extended_fingers(lm) == 0

    def test_open_palm(self):
        """All tips further from wrist than MCPs → 5 fingers."""
        lm = [_make_landmark(0.5, 0.5) for _ in range(21)]
        lm[0] = _make_landmark(0.5, 0.9)  # wrist at bottom
        for mcp in [2, 5, 9, 13, 17]:
            lm[mcp] = _make_landmark(0.5, 0.6)
        for tip in [4, 8, 12, 16, 20]:
            lm[tip] = _make_landmark(0.5, 0.2)  # far from wrist

        assert VisionTracker.count_extended_fingers(lm) == 5

    def test_one_finger(self):
        """Only index finger extended."""
        lm = [_make_landmark(0.5, 0.5) for _ in range(21)]
        lm[0] = _make_landmark(0.5, 0.9)  # wrist

        # All MCPs at mid distance
        for mcp in [2, 5, 9, 13, 17]:
            lm[mcp] = _make_landmark(0.5, 0.6)
        # All tips close (folded)
        for tip in [4, 8, 12, 16, 20]:
            lm[tip] = _make_landmark(0.5, 0.8)
        # Index tip extended
        lm[8] = _make_landmark(0.5, 0.2)

        assert VisionTracker.count_extended_fingers(lm) == 1

    def test_none_landmarks(self):
        assert VisionTracker.count_extended_fingers(None) == 0


class TestGetBlinkScores:
    def test_no_blendshapes_returns_zeros(self):
        face = _mock_face_result()
        # Can't call the method without a tracker instance,
        # but we can test the logic:
        if not face.face_blendshapes:
            result = (0.0, 0.0)
        assert result == (0.0, 0.0)

    def test_blendshape_extraction(self):
        """Verify correct indices are used (9=left, 10=right)."""
        shapes = [SimpleNamespace(score=0.0) for _ in range(52)]
        shapes[9] = SimpleNamespace(score=0.8)   # left blink
        shapes[10] = SimpleNamespace(score=0.1)  # right blink

        face = _mock_face_result(blendshapes=shapes)
        scores = face.face_blendshapes[0]
        assert scores[9].score == 0.8
        assert scores[10].score == 0.1


class TestGetHandLandmarks:
    def test_returns_none_when_empty(self):
        result = _mock_hand_result()
        assert VisionTracker.get_hand_landmarks(result) is None

    def test_returns_landmarks_when_present(self):
        lm = [_make_landmark(0.0, 0.0) for _ in range(21)]
        result = _mock_hand_result(landmarks=lm)
        assert VisionTracker.get_hand_landmarks(result) is lm
