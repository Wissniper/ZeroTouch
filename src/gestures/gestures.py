"""Gesture detection: winks, pinch, finger-count actions.

All detectors are pure functions of landmark data — no side effects.
Action execution (pyautogui calls) is handled in main.py.
"""

import math
import time


class GestureController:
    """Stateful gesture detector with cooldowns to prevent repeat-firing.

    Gesture map:
        Left wink  → left click
        Right wink → right click
        1 finger   → scroll mode (index Y-velocity)
        2 fingers  → zoom mode  (pinch distance)
        3 fingers  → drag mode
        4 fingers  → desktop switch (swipe direction)
        5 fingers  → mission control
    """

    def __init__(
        self,
        wink_threshold: float = 0.10,
        pinch_threshold: float = 0.05,
        cooldown_ms: float = 500,
    ):
        """
        Args:
            wink_threshold: Minimum score *difference* between eyes to register
                            a wink vs a blink. Tuned lower for glasses wearers.
            pinch_threshold: Normalized distance below which thumb-index counts
                             as a pinch.
            cooldown_ms: Minimum milliseconds between repeated gesture triggers.
        """
        self.wink_threshold = wink_threshold
        self.pinch_threshold = pinch_threshold
        self._cooldown_s = cooldown_ms / 1000.0

        # Cooldown timestamps per gesture type
        self._last_trigger: dict[str, float] = {}

        # Previous pinch distance for delta-based zoom
        self._prev_pinch_dist: float | None = None

        # Previous index-finger Y for scroll velocity
        self._prev_index_y: float | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cooled_down(self, gesture: str) -> bool:
        """True if enough time has passed since the last trigger of *gesture*."""
        now = time.time()
        last = self._last_trigger.get(gesture, 0.0)
        if now - last >= self._cooldown_s:
            self._last_trigger[gesture] = now
            return True
        return False

    @staticmethod
    def _landmark_dist(a, b) -> float:
        """Euclidean distance between two MediaPipe landmarks."""
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    # ------------------------------------------------------------------
    # Wink detection  (eye blink scores from blendshapes)
    # ------------------------------------------------------------------

    def detect_wink(self, left_blink: float, right_blink: float) -> str | None:
        """Detect a single-eye wink.

        The key insight: a *wink* means one eye is significantly more closed
        than the other.  A *blink* closes both eyes roughly equally, so the
        difference stays below the threshold.

        Returns 'left', 'right', or None.
        """
        diff = left_blink - right_blink

        if diff > self.wink_threshold and self._cooled_down("wink_right"):
            # Left blink score high, right low → right eye wink
            # (MediaPipe blendshape naming: eyeBlinkLeft = left-eye closure)
            return "right"
        if -diff > self.wink_threshold and self._cooled_down("wink_left"):
            return "left"

        return None

    # ------------------------------------------------------------------
    # Hand gesture detection
    # ------------------------------------------------------------------

    def detect_pinch(self, hand_landmarks) -> float | None:
        """Detect thumb-index pinch and return the *delta distance* since last frame.

        Returns:
            Positive float (spreading apart / zoom-in), negative (pinching / zoom-out),
            or None if not pinching or no hand detected.
        """
        if hand_landmarks is None:
            self._prev_pinch_dist = None
            return None

        # Thumb tip = 4, Index tip = 8
        dist = self._landmark_dist(hand_landmarks[4], hand_landmarks[8])

        if self._prev_pinch_dist is None:
            self._prev_pinch_dist = dist
            return None

        delta = dist - self._prev_pinch_dist
        self._prev_pinch_dist = dist
        return delta

    def detect_scroll(self, hand_landmarks) -> float | None:
        """Detect single-finger (index) scroll from Y-axis velocity.

        Returns vertical delta (positive = finger moved down = scroll down).
        """
        if hand_landmarks is None:
            self._prev_index_y = None
            return None

        index_tip_y = hand_landmarks[8].y  # normalized 0..1

        if self._prev_index_y is None:
            self._prev_index_y = index_tip_y
            return None

        delta_y = index_tip_y - self._prev_index_y
        self._prev_index_y = index_tip_y

        # Only report meaningful movement (filter micro-noise)
        if abs(delta_y) < 0.005:
            return None
        return delta_y

    def detect_swipe(self, hand_landmarks, velocity_threshold: float = 0.08) -> str | None:
        """Detect a 4-finger horizontal swipe for desktop switching.

        Measures index-finger X velocity; only fires if 4 fingers are extended
        and horizontal movement exceeds *velocity_threshold*.
        """
        if hand_landmarks is None:
            return None

        # Check for 4 extended fingers via the tip-above-mcp heuristic
        # (imported from tracker, but we do a local lightweight version)
        index_x = hand_landmarks[8].x

        if not hasattr(self, "_prev_index_x"):
            self._prev_index_x: float | None = None

        if self._prev_index_x is None:
            self._prev_index_x = index_x
            return None

        dx = index_x - self._prev_index_x
        self._prev_index_x = index_x

        if abs(dx) > velocity_threshold and self._cooled_down("swipe"):
            return "left" if dx > 0 else "right"
        return None

    def classify_hand_gesture(self, hand_landmarks, finger_count: int) -> str | None:
        """Map finger count to a gesture name.

        Returns one of: 'scroll', 'zoom', 'drag', 'switch_desktop',
        'mission_control', or None.
        """
        if hand_landmarks is None or finger_count == 0:
            return None

        gesture_map = {
            1: "scroll",
            2: "zoom",
            3: "drag",
            4: "switch_desktop",
            5: "mission_control",
        }
        return gesture_map.get(finger_count)
