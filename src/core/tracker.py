import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import cv2 as cv
import numpy as np

class VisionTracker:
    def __init__(self, face_model_path='face_landmarker.task', hand_model_path='hand_landmarker.task'):
        # Setup Face Landmarker
        base_face = python.BaseOptions(model_asset_path=face_model_path)
        self.face_detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base_face,
                output_face_blendshapes=True,
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
