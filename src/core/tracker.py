import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv
import numpy as np
import os
import math

_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..')

# Small constant to prevent division by zero in gaze ratio calculations.
# Eye-socket width is typically ~0.05 in normalized coords, so 1e-6 is safe.
_EPS = 1e-6


class VisionTracker:
    """Wraps MediaPipe Face + Hand Landmarkers into a single detection API.

    Raises FileNotFoundError at init if model .task files are missing
    (run ``python setup_models.py`` first).
    """

    def __init__(self, face_model_path=None, hand_model_path=None):
        if face_model_path is None:
            face_model_path = os.path.join(_MODELS_DIR, 'face_landmarker.task')
        if hand_model_path is None:
            hand_model_path = os.path.join(_MODELS_DIR, 'hand_landmarker.task')

        for path, name in [(face_model_path, "face"), (hand_model_path, "hand")]:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"{name} model not found at {path}. Run: python setup_models.py"
                )

        # Face Landmarker — blendshapes for blink, transformation matrix for head pose
        base_face = python.BaseOptions(model_asset_path=face_model_path)
        self.face_detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base_face,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1,
            )
        )

        # Hand Landmarker — 21 landmarks per hand
        base_hand = python.BaseOptions(model_asset_path=hand_model_path)
        self.hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base_hand,
                num_hands=1,
            )
        )

    def process(self, frame):
        """Run face + hand detection on a BGR frame.

        Returns:
            (face_result, hand_result) — MediaPipe detection result objects.
        """
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.face_detector.detect(mp_image), self.hand_detector.detect(mp_image)

    # ------------------------------------------------------------------
    # Iris & Gaze
    # ------------------------------------------------------------------

    def get_iris_coords(self, face_result):
        """Average normalized (x, y) of all 10 iris landmarks (468-477)."""
        if not face_result.face_landmarks:
            return None
        marks = face_result.face_landmarks[0]
        indices = list(range(468, 478))
        return (
            sum(marks[i].x for i in indices) / len(indices),
            sum(marks[i].y for i in indices) / len(indices),
        )

    def get_gaze_ratio(self, face_result):
        """Iris position relative to the eye socket.

        Returns (x_ratio, y_ratio) where 0.5 ≈ centered gaze.
        Uses _EPS to guard against division-by-zero when the eye-socket
        landmarks collapse (e.g. during a blink or extreme angle).
        """
        if not face_result.face_landmarks:
            return None
        marks = face_result.face_landmarks[0]

        # Left Eye (33=outer, 133=inner, 159=top, 145=bottom)
        # Right Eye (362=inner, 263=outer, 386=top, 374=bottom)
        l_iris, r_iris = marks[468], marks[473]

        # Horizontal ratio  (0.0 = looking left, 1.0 = looking right)
        l_h = (l_iris.x - marks[33].x) / (marks[133].x - marks[33].x + _EPS)
        r_h = (r_iris.x - marks[362].x) / (marks[263].x - marks[362].x + _EPS)

        # Vertical ratio  (0.0 = looking up, 1.0 = looking down)
        l_v = (l_iris.y - marks[159].y) / (marks[145].y - marks[159].y + _EPS)
        r_v = (r_iris.y - marks[386].y) / (marks[374].y - marks[386].y + _EPS)

        avg_h = (l_h + r_h) / 2.0
        avg_v = 1.0 - (l_v + r_v) / 2.0

        return avg_h, avg_v

    # ------------------------------------------------------------------
    # Blink / Head Pose
    # ------------------------------------------------------------------

    def get_blink_scores(self, face_result):
        """Return (left_blink, right_blink) from blendshapes. 0.0 if unavailable."""
        if not face_result.face_blendshapes:
            return 0.0, 0.0
        shapes = face_result.face_blendshapes[0]
        # Index 9 = eyeBlinkLeft, 10 = eyeBlinkRight
        return shapes[9].score, shapes[10].score

    def get_head_pose(self, face_result):
        """Extract (yaw, pitch) from the facial transformation matrix.

        matrix[0,2] ≈ sin(yaw)   — positive when face turns right in camera
        matrix[1,2] ≈ sin(pitch) — positive when face tilts down
        """
        if not face_result.facial_transformation_matrixes:
            return None
        matrix = face_result.facial_transformation_matrixes[0]
        yaw = float(matrix[0, 2])
        pitch = float(matrix[1, 2])
        return (yaw, pitch)

    # ------------------------------------------------------------------
    # Hand helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_hand_landmarks(hand_result):
        """Return the first hand's landmark list, or None."""
        if hand_result.hand_landmarks:
            return hand_result.hand_landmarks[0]
        return None

    @staticmethod
    def count_extended_fingers(landmarks) -> int:
        """Count how many fingers are extended (roughly open).

        Uses a simple heuristic: a finger is extended when its tip landmark
        is further from the wrist (landmark 0) than its MCP joint.

        Landmark indices per finger:
            Thumb:  tip=4,  mcp=1
            Index:  tip=8,  mcp=5
            Middle: tip=12, mcp=9
            Ring:   tip=16, mcp=13
            Pinky:  tip=20, mcp=17
        """
        if landmarks is None:
            return 0

        tip_ids = [4, 8, 12, 16, 20]
        finger_states = []
        for tip_id in tip_ids:
            finger_tip = landmarks[tip_id]
            finger_mcp = landmarks[tip_id - 3]
            # Check if finger is open or closed
            if tip_id==4:
                finger_states.append(finger_tip.x < finger_mcp.x)
            else:
                finger_states.append(finger_tip.y < finger_mcp.y)
        # Count number of open fingers
        count = finger_states.count(True)

        return count
