"""Tests for GestureController — wink, pinch, scroll, swipe, classify."""

import time
from types import SimpleNamespace
import pytest
from src.gestures.gestures import GestureController


def _make_landmark(x, y, z=0.0):
    """Create a mock MediaPipe landmark (just needs .x, .y, .z)."""
    return SimpleNamespace(x=x, y=y, z=z)


def _make_hand(positions: dict[int, tuple[float, float]]):
    """Build a sparse hand-landmark list from {index: (x,y)} dict.

    Fills all 21 slots; missing indices default to (0.5, 0.5).
    """
    lm = [_make_landmark(0.5, 0.5) for _ in range(21)]
    for idx, (x, y) in positions.items():
        lm[idx] = _make_landmark(x, y)
    return lm


class TestWinkDetection:
    def test_no_wink_when_both_eyes_similar(self):
        gc = GestureController(wink_threshold=0.10, cooldown_ms=0)
        assert gc.detect_wink(0.05, 0.05) is None
        assert gc.detect_wink(0.50, 0.50) is None

    def test_left_wink(self):
        gc = GestureController(wink_threshold=0.10, cooldown_ms=0)
        # right_blink much higher than left → left wink
        result = gc.detect_wink(0.02, 0.40)
        assert result == "left"

    def test_right_wink(self):
        gc = GestureController(wink_threshold=0.10, cooldown_ms=0)
        # left_blink much higher than right → right wink
        result = gc.detect_wink(0.40, 0.02)
        assert result == "right"

    def test_cooldown_prevents_rapid_fire(self):
        gc = GestureController(wink_threshold=0.10, cooldown_ms=1000)
        # First wink should fire
        assert gc.detect_wink(0.40, 0.02) == "right"
        # Immediate second should be blocked by cooldown
        assert gc.detect_wink(0.40, 0.02) is None

    def test_blink_rejected(self):
        """Both eyes closing equally should not trigger a wink."""
        gc = GestureController(wink_threshold=0.10, cooldown_ms=0)
        # Both high (blink) — difference is small
        assert gc.detect_wink(0.80, 0.80) is None
        assert gc.detect_wink(0.80, 0.75) is None


class TestPinchDetection:
    def test_no_hand_returns_none(self):
        gc = GestureController(cooldown_ms=0)
        assert gc.detect_pinch(None) is None

    def test_first_frame_returns_none(self):
        """First frame has no previous distance to compare."""
        gc = GestureController(cooldown_ms=0)
        hand = _make_hand({4: (0.5, 0.5), 8: (0.6, 0.6)})
        assert gc.detect_pinch(hand) is None

    def test_pinch_delta_positive_when_spreading(self):
        gc = GestureController(cooldown_ms=0)
        # Frame 1: fingers close
        hand1 = _make_hand({4: (0.5, 0.5), 8: (0.55, 0.55)})
        gc.detect_pinch(hand1)

        # Frame 2: fingers spread
        hand2 = _make_hand({4: (0.5, 0.5), 8: (0.8, 0.8)})
        delta = gc.detect_pinch(hand2)
        assert delta is not None
        assert delta > 0

    def test_pinch_delta_negative_when_closing(self):
        gc = GestureController(cooldown_ms=0)
        hand1 = _make_hand({4: (0.5, 0.5), 8: (0.8, 0.8)})
        gc.detect_pinch(hand1)

        hand2 = _make_hand({4: (0.5, 0.5), 8: (0.55, 0.55)})
        delta = gc.detect_pinch(hand2)
        assert delta is not None
        assert delta < 0

    def test_reset_on_hand_loss(self):
        gc = GestureController(cooldown_ms=0)
        hand = _make_hand({4: (0.5, 0.5), 8: (0.6, 0.6)})
        gc.detect_pinch(hand)
        gc.detect_pinch(None)  # hand lost

        # Next frame should act as first frame again
        assert gc.detect_pinch(hand) is None


class TestScrollDetection:
    def test_no_hand_returns_none(self):
        gc = GestureController(cooldown_ms=0)
        assert gc.detect_scroll(None) is None

    def test_first_frame_returns_none(self):
        gc = GestureController(cooldown_ms=0)
        hand = _make_hand({8: (0.5, 0.5)})
        assert gc.detect_scroll(hand) is None

    def test_detects_downward_scroll(self):
        gc = GestureController(cooldown_ms=0)
        gc.detect_scroll(_make_hand({8: (0.5, 0.3)}))
        delta = gc.detect_scroll(_make_hand({8: (0.5, 0.5)}))
        assert delta is not None
        assert delta > 0

    def test_ignores_micro_noise(self):
        gc = GestureController(cooldown_ms=0)
        gc.detect_scroll(_make_hand({8: (0.5, 0.5000)}))
        delta = gc.detect_scroll(_make_hand({8: (0.5, 0.5001)}))
        assert delta is None  # below 0.005 threshold


class TestSwipeDetection:
    def test_no_hand_returns_none(self):
        gc = GestureController(cooldown_ms=0)
        assert gc.detect_swipe(None) is None

    def test_detects_left_swipe(self):
        gc = GestureController(cooldown_ms=0)
        gc.detect_swipe(_make_hand({8: (0.3, 0.5)}))
        result = gc.detect_swipe(_make_hand({8: (0.5, 0.5)}))
        assert result == "left"  # dx > 0 in camera coords

    def test_detects_right_swipe(self):
        gc = GestureController(cooldown_ms=0)
        gc.detect_swipe(_make_hand({8: (0.7, 0.5)}))
        result = gc.detect_swipe(_make_hand({8: (0.5, 0.5)}))
        assert result == "right"


class TestClassifyHandGesture:
    def test_no_hand_returns_none(self):
        gc = GestureController()
        assert gc.classify_hand_gesture(None, 0) is None

    def test_zero_fingers_returns_none(self):
        gc = GestureController()
        hand = _make_hand({})
        assert gc.classify_hand_gesture(hand, 0) is None

    @pytest.mark.parametrize("count,expected", [
        (1, "scroll"),
        (2, "zoom"),
        (3, "drag"),
        (4, "switch_desktop"),
        (5, "mission_control"),
    ])
    def test_finger_count_mapping(self, count, expected):
        gc = GestureController()
        hand = _make_hand({})
        assert gc.classify_hand_gesture(hand, count) == expected
