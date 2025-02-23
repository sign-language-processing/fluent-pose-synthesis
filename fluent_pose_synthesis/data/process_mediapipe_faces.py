from typing import Any, Dict, List, Optional, Set
from mediapipe.python.solutions.face_mesh_connections import (
    FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_RIGHT_EYE,
    FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL
)


# Face Regions Mapping (based on MediaPipe definitions)
MEDIAPIPE_FACIAL_REGIONS = {
    "lips": FACEMESH_LIPS,
    "left_eye": FACEMESH_LEFT_EYE,
    "right_eye": FACEMESH_RIGHT_EYE,
    "left_eyebrow": FACEMESH_LEFT_EYEBROW,
    "right_eyebrow": FACEMESH_RIGHT_EYEBROW,
    "face_oval": FACEMESH_FACE_OVAL
}

# MediaPipe Face Key Points Mapping
MEDIAPIPE_FACE_KEYPOINTS = {
    "upper_lip_center": 13,
    "lower_lip_center": 14,
    "upper_lip_left": 82,
    "lower_lip_left": 87,
    "upper_lip_right": 312,
    "lower_lip_right": 317,
    "lip_left_corner": 78,
    "lip_right_corner": 308,
    "left_eye_upper": 386,
    "left_eye_lower": 374,
    "right_eye_upper": 159,
    "right_eye_lower": 145
}


def extract_mediapipe_face_regions() -> Dict[str, Set[int]]:
    """
    Extract face landmark indices for each facial region.
    """
    return {region: set().union(*connections) for region, connections in MEDIAPIPE_FACIAL_REGIONS.items()}


mediapipe_face_regions_dict = extract_mediapipe_face_regions()