"""MediaPipe Pose wrapper: raw frame -> PoseFrame, restricted to our landmark subset."""
from __future__ import annotations

import numpy as np

from src.pose_estimation.landmark_schema import (
    MEDIAPIPE_INDEX,
    Landmark,
    LandmarkName,
    PoseFrame,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PoseEstimator:
    """Wraps `mediapipe.solutions.pose.Pose`.

    MediaPipe's `solutions.*` API (used here, pinned to mediapipe==0.10.14) performs
    person detection and landmark estimation in one call - there is no separate detector
    step to invoke for this subset of landmarks.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        landmark_confidence_threshold: float = 0.5,
        static_image_mode: bool = False,
    ):
        import mediapipe as mp  # imported lazily so unit tests can mock this module

        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmark_confidence_threshold = landmark_confidence_threshold

    def estimate(self, image_bgr: np.ndarray, frame_index: int, timestamp_sec: float) -> PoseFrame:
        import cv2

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(image_rgb)

        if result.pose_landmarks is None:
            return PoseFrame(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                detected=False,
                landmarks={name: Landmark(None, None, None, 0.0) for name in LandmarkName},
            )

        raw_landmarks = result.pose_landmarks.landmark
        landmarks: dict[LandmarkName, Landmark] = {}
        for name, mp_idx in MEDIAPIPE_INDEX.items():
            lm = raw_landmarks[mp_idx]
            if lm.visibility < self.landmark_confidence_threshold:
                landmarks[name] = Landmark(None, None, None, lm.visibility)
            else:
                landmarks[name] = Landmark(lm.x, lm.y, lm.z, lm.visibility)

        return PoseFrame(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            detected=True,
            landmarks=landmarks,
        )

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
