"""Classic joint angles (elbow, knee, hip) computed via the law of cosines.

These are generic biomechanical angles, independent of any posture judgment - 180 degrees
means a fully extended joint, smaller values mean more flexion. Posture-specific
interpretation (e.g. "is this knee angle a deep squat") lives in posture_analysis, which
consumes these values alongside configurable thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.feature_extraction.geometry import angle_at_vertex, to_point
from src.pose_estimation.landmark_schema import LandmarkName, PoseFrame


@dataclass
class JointAngles:
    left_elbow_deg: float | None
    right_elbow_deg: float | None
    left_knee_deg: float | None
    right_knee_deg: float | None
    left_hip_deg: float | None
    right_hip_deg: float | None


def _safe_angle(pose_frame: PoseFrame, a: LandmarkName, b: LandmarkName, c: LandmarkName) -> float | None:
    lm = pose_frame.landmarks
    if not (lm[a].is_present and lm[b].is_present and lm[c].is_present):
        return None
    try:
        return angle_at_vertex(to_point(lm[a]), to_point(lm[b]), to_point(lm[c]))
    except ValueError:
        return None


def compute_joint_angles(pose_frame: PoseFrame) -> JointAngles:
    return JointAngles(
        left_elbow_deg=_safe_angle(
            pose_frame, LandmarkName.LEFT_SHOULDER, LandmarkName.LEFT_ELBOW, LandmarkName.LEFT_WRIST
        ),
        right_elbow_deg=_safe_angle(
            pose_frame, LandmarkName.RIGHT_SHOULDER, LandmarkName.RIGHT_ELBOW, LandmarkName.RIGHT_WRIST
        ),
        left_knee_deg=_safe_angle(
            pose_frame, LandmarkName.LEFT_HIP, LandmarkName.LEFT_KNEE, LandmarkName.LEFT_ANKLE
        ),
        right_knee_deg=_safe_angle(
            pose_frame, LandmarkName.RIGHT_HIP, LandmarkName.RIGHT_KNEE, LandmarkName.RIGHT_ANKLE
        ),
        left_hip_deg=_safe_angle(
            pose_frame, LandmarkName.LEFT_SHOULDER, LandmarkName.LEFT_HIP, LandmarkName.LEFT_KNEE
        ),
        right_hip_deg=_safe_angle(
            pose_frame, LandmarkName.RIGHT_SHOULDER, LandmarkName.RIGHT_HIP, LandmarkName.RIGHT_KNEE
        ),
    )
