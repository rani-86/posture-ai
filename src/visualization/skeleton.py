"""Draws the pose skeleton (joints + bones) over a video frame for QA and reporting."""
from __future__ import annotations

import cv2
import numpy as np

from src.pose_estimation.landmark_schema import SKELETON_EDGES, LandmarkName, PoseFrame

JOINT_COLOR = (0, 215, 255)  # amber, BGR
BONE_COLOR = (80, 220, 80)  # green, BGR
MISSING_JOINT_COLOR = (60, 60, 220)  # red, BGR - flags held/missing landmarks


def draw_skeleton(image_bgr: np.ndarray, pose_frame: PoseFrame, radius: int = 4) -> np.ndarray:
    """Returns a copy of `image_bgr` with the skeleton drawn on it. Does not mutate input."""
    annotated = image_bgr.copy()
    height, width = annotated.shape[:2]

    def to_pixel(name: LandmarkName) -> tuple[int, int] | None:
        lm = pose_frame.landmarks[name]
        if not lm.is_present:
            return None
        return int(lm.x * width), int(lm.y * height)

    for start, end in SKELETON_EDGES:
        p1, p2 = to_pixel(start), to_pixel(end)
        if p1 is not None and p2 is not None:
            cv2.line(annotated, p1, p2, BONE_COLOR, thickness=2)

    for name in LandmarkName:
        p = to_pixel(name)
        if p is None:
            continue
        lm = pose_frame.landmarks[name]
        # visibility == 0 with coordinates present means the smoother held a stale value.
        color = MISSING_JOINT_COLOR if lm.visibility == 0.0 else JOINT_COLOR
        cv2.circle(annotated, p, radius, color, thickness=-1)

    return annotated
