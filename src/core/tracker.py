import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import cv2 as cv
import numpy as np
import os

_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..')

class VisionTracker:
    def __init__(self, face_model_path=None, hand_model_path=None):
        if face_model_path is None:
            face_model_path = os.path.join(_MODELS_DIR, 'face_landmarker.task')
        if hand_model_path is None:
            hand_model_path = os.path.join(_MODELS_DIR, 'hand_landmarker.task')
        # Setup Face Landmarker
        base_face = python.BaseOptions(model_asset_path=face_model_path)
        self.face_detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base_face,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1
            )
        )

        # Setup Hand Landmarker
        base_hand = python.BaseOptions(model_asset_path=hand_model_path)
        self.hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base_hand,
                num_hands=1 
            )
        )

    def process(self, frame):
        """Processes RGB frame for landmarks."""
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        return self.face_detector.detect(mp_image), self.hand_detector.detect(mp_image)

    def get_iris_coords(self, face_result):
        """Extracts normalized (x, y) for the average iris position."""
        if not face_result.face_landmarks: return None
        marks = face_result.face_landmarks[0]
        # Indices for irises
        indices = list(range(468, 478))
        return (sum(marks[i].x for i in indices) / 10, sum(marks[i].y for i in indices) / 10)

    def get_blink_scores(self, face_result):
        """Returns (left, right) blink scores."""
        if not face_result.face_blendshapes: return 0.0, 0.0
        shapes = face_result.face_blendshapes[0]
        # 9: left, 10: right
        return shapes[9].score, shapes[10].score

    def get_head_pose(self, face_result):
        """
        Extracts head rotation from the transformation matrix.
        Returns (yaw, pitch) as rotation matrix elements (approx. sin of angle, range -1..1).

        matrix[0, 2] = horizontal component of the face's forward (nose) direction in camera space.
                       Positive when the face turns RIGHT in the (flipped) image, i.e. user turns LEFT.
        matrix[1, 2] = vertical component of the same vector.
                       Positive when the face tilts DOWN.

        These are used for head compensation:
            comp_x = gaze_ratio[0] - yaw  * HEAD_COMP_SCALE
            comp_y = gaze_ratio[1] - pitch * HEAD_COMP_SCALE
        """
        if not face_result.facial_transformation_matrixes: return None
        matrix = face_result.facial_transformation_matrixes[0]

        yaw   = matrix[0, 2]   # ~sin(horizontal turn)
        pitch = matrix[1, 2]   # ~sin(vertical tilt)

        return (yaw, pitch)

    def get_gaze_ratio(self, face_result):
        """
        Calculates the iris position relative to the eye socket.
        Returns (x_ratio, y_ratio) where 0.5 is centered gaze.
        """
        if not face_result.face_landmarks: return None
        marks = face_result.face_landmarks[0]
        
        # Left Eye (33=outer, 133=inner, 159=top, 145=bottom)
        # Right Eye (362=inner, 263=outer, 386=top, 374=bottom)
        l_iris, r_iris = marks[468], marks[473]

        # Horizontal ratio (0.0 = looking left, 1.0 = looking right)
        # Note: We use the flipped logic since camera is mirrored
        l_h = (l_iris.x - marks[33].x) / (marks[133].x - marks[33].x)
        r_h = (r_iris.x - marks[362].x) / (marks[263].x - marks[362].x)
        
        # Vertical ratio (0.0 = looking up, 1.0 = looking down)
        l_v = (l_iris.y - marks[159].y) / (marks[145].y - marks[159].y)
        r_v = (r_iris.y - marks[386].y) / (marks[374].y - marks[386].y)
        
        # Average both eyes and invert horizontal for intuitive mirroring
        avg_h = ((l_h + r_h) / 2.0)
        avg_v = 1.0 - (l_v + r_v) / 2.0
        
        return avg_h, avg_v
