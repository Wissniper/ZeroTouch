import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import cv2 as cv
import numpy as np

class VisionTracker:
    def __init__(self, face_model_path='face_landmarker.task', hand_model_path='hand_landmarker.task'):
        """
        Initializes Face and Hand landmarker models once for performance.
        Ensure the .task files are in the root directory.
        """
        # Face Landmarker Setup
        base_options_face = python.BaseOptions(model_asset_path=face_model_path)
        options_face = vision.FaceLandmarkerOptions(
            base_options=base_options_face,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.face_detector = vision.FaceLandmarker.create_from_options(options_face)

        # Hand Landmarker Setup
        base_options_hand = python.BaseOptions(model_asset_path=hand_model_path)
        options_hand = vision.HandLandmarkerOptions(
            base_options=base_options_hand,
            num_hands=2
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(options_hand)

    def process(self, frame):
        """Processes a single BGR frame and returns both face and hand landmarks."""
        # MediaPipe expects RGB
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        face_result = self.face_detector.detect(mp_image)
        hand_result = self.hand_detector.detect(mp_image)

        return face_result, hand_result

def draw_face_landmarks(rgb_image, detection_result):
    """Utility to draw face and iris landmarks on an image."""
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
        
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

    return annotated_image

def draw_hand_landmarks(rgb_image, detection_result):
    """Utility to draw hand landmarks on an image."""
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=hand_landmarks,
            connections=vision.HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=drawing_styles.get_default_hand_connections_style())

    return annotated_image
