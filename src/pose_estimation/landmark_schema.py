"""Landmark naming/schema shared across pose estimation, features, and visualization.

MediaPipe Pose returns 33 landmarks; this project only needs the subset relevant to
posture and joint-angle analysis. Keeping an explicit subset (rather than passing all 33
through every downstream module) makes feature extraction and visualization code easier
to reason about and keeps missing/low-confidence handling scoped to what we actually use.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LandmarkName(str, Enum):
    NOSE = "nose"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"


# Maps our schema to MediaPipe's PoseLandmark enum indices (0-32).
# See https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#pose_landmarker_model
MEDIAPIPE_INDEX = {
    LandmarkName.NOSE: 0,
    LandmarkName.LEFT_EAR: 7,
    LandmarkName.RIGHT_EAR: 8,
    LandmarkName.LEFT_SHOULDER: 11,
    LandmarkName.RIGHT_SHOULDER: 12,
    LandmarkName.LEFT_ELBOW: 13,
    LandmarkName.RIGHT_ELBOW: 14,
    LandmarkName.LEFT_WRIST: 15,
    LandmarkName.RIGHT_WRIST: 16,
    LandmarkName.LEFT_HIP: 23,
    LandmarkName.RIGHT_HIP: 24,
    LandmarkName.LEFT_KNEE: 25,
    LandmarkName.RIGHT_KNEE: 26,
    LandmarkName.LEFT_ANKLE: 27,
    LandmarkName.RIGHT_ANKLE: 28,
}

# Bone connections among the subset above, used for skeleton drawing.
SKELETON_EDGES: list[tuple[LandmarkName, LandmarkName]] = [
    (LandmarkName.NOSE, LandmarkName.LEFT_EAR),
    (LandmarkName.NOSE, LandmarkName.RIGHT_EAR),
    (LandmarkName.LEFT_SHOULDER, LandmarkName.RIGHT_SHOULDER),
    (LandmarkName.LEFT_SHOULDER, LandmarkName.LEFT_ELBOW),
    (LandmarkName.LEFT_ELBOW, LandmarkName.LEFT_WRIST),
    (LandmarkName.RIGHT_SHOULDER, LandmarkName.RIGHT_ELBOW),
    (LandmarkName.RIGHT_ELBOW, LandmarkName.RIGHT_WRIST),
    (LandmarkName.LEFT_SHOULDER, LandmarkName.LEFT_HIP),
    (LandmarkName.RIGHT_SHOULDER, LandmarkName.RIGHT_HIP),
    (LandmarkName.LEFT_HIP, LandmarkName.RIGHT_HIP),
    (LandmarkName.LEFT_HIP, LandmarkName.LEFT_KNEE),
    (LandmarkName.LEFT_KNEE, LandmarkName.LEFT_ANKLE),
    (LandmarkName.RIGHT_HIP, LandmarkName.RIGHT_KNEE),
    (LandmarkName.RIGHT_KNEE, LandmarkName.RIGHT_ANKLE),
]


@dataclass
class Landmark:
    """A single landmark in normalized image coordinates ([0,1] x/y) plus depth and
    MediaPipe's visibility score. `None` x/y/z means the landmark was filtered out for
    being below the confidence threshold or because no person was detected."""

    x: float | None
    y: float | None
    z: float | None
    visibility: float

    @property
    def is_present(self) -> bool:
        return self.x is not None and self.y is not None


@dataclass
class PoseFrame:
    """Pose estimation result for a single video frame."""

    frame_index: int
    timestamp_sec: float
    detected: bool
    landmarks: dict[LandmarkName, Landmark]

    def get(self, name: LandmarkName) -> Landmark:
        return self.landmarks[name]
