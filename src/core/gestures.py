class GestureController:
    def __init__(self, wink_threshold=0.10):
        """
        Handles state detection for eye winks and hand gestures.
        
        Args:
            wink_threshold (float): The sensitivity for a wink (adjusted for glasses).
            open_threshold (float): The sensitivity for an open eye.
        """
        self.wink_threshold = wink_threshold

    def detect_wink(self, left_blink, right_blink):
        """
        Detects if a single eye is closed based on relative scores.
        Returns: 'left', 'right', or None
        """
        # Given your glasses, we use a much lower threshold
        is_right_closed = left_blink - right_blink > self.wink_threshold
        is_left_closed = right_blink - left_blink > self.wink_threshold

        # Logic for a single-eye wink
        if is_left_closed:
            return 'left'
        elif is_right_closed:
            return 'right'
        
        return None

    def detect_hand_pinch(self, hand_landmarks):
        """
        Placeholder for hand gesture logic (Zoom/Click).
        Will use Euclidean distance between Thumb (4) and Index (8).
        """
        if not hand_landmarks:
            return False
        # Future implementation here...
        return False
