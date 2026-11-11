"""Posture-specific geometric features: head alignment, shoulder/hip symmetry, trunk and
body inclination. These are deterministic, rule-based measurements - no ML involved -
built on top of the joint angles and raw landmarks from Phase 2/3.

Camera-angle assumptions (important, see docs/phase3_notes.md for full discussion):
- `forward_head_angle_deg` approximates the clinical "craniovertebral angle" (CVA) used
  in physiotherapy to assess forward head posture. CVA is only meaningful from a side
  (sagittal-plane) view of the subject; from a front-facing camera the ear-to-shoulder
  horizontal offset this relies on is compressed to near zero and the measurement is not
  reliable. This is a geometric limitation of using one RGB camera, not a bug.
- `shoulder_tilt_deg`, `hip_tilt_deg`, `trunk_inclination_deg`, `body_inclination_deg` are
  most meaningful from a front-facing view (they measure left-right asymmetry/lean) and
  are less reliable from a pure side view, where left/right landmarks overlap.
No single video angle supports all of these measurements equally well; this is why
thresholds are configurable per deployment context rather than fixed constants.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.feature_extraction.geometry import angle_from_horizontal, angle_from_vertical, midpoint, to_point, vector
from src.feature_extraction.joint_angles import JointAngles, compute_joint_angles
from src.pose_estimation.landmark_schema import LandmarkName, PoseFrame


@dataclass
class PostureFeatures:
    forward_head_angle_deg: float | None  # approximate CVA; smaller = more forward head
    shoulder_tilt_deg: float | None  # 0 = level shoulders
    hip_tilt_deg: float | None  # 0 = level hips
    trunk_inclination_deg: float | None  # 0 = upright trunk (hip-to-shoulder line)
    body_inclination_deg: float | None  # 0 = upright whole body (ankle-to-shoulder line)
    left_knee_deg: float | None
    right_knee_deg: float | None
    joint_angles: JointAngles


def _best_side_ear_shoulder(pose_frame: PoseFrame) -> tuple[LandmarkName, LandmarkName] | None:
    """Picks whichever side (left/right) has both ear and shoulder visible, preferring
    the side with higher combined visibility - relevant for a side-view video where only
    one side of the body faces the camera."""
    lm = pose_frame.landmarks
    candidates = [
        (LandmarkName.LEFT_EAR, LandmarkName.LEFT_SHOULDER),
        (LandmarkName.RIGHT_EAR, LandmarkName.RIGHT_SHOULDER),
    ]
    best = None
    best_score = -1.0
    for ear, shoulder in candidates:
        if lm[ear].is_present and lm[shoulder].is_present:
            score = lm[ear].visibility + lm[shoulder].visibility
            if score > best_score:
                best, best_score = (ear, shoulder), score
    return best


def compute_posture_features(pose_frame: PoseFrame) -> PostureFeatures:
    lm = pose_frame.landmarks
    joint_angles = compute_joint_angles(pose_frame)

    forward_head_angle_deg = None
    side = _best_side_ear_shoulder(pose_frame)
    if side is not None:
        ear_name, shoulder_name = side
        forward_head_angle_deg = angle_from_horizontal(vector(to_point(lm[shoulder_name]), to_point(lm[ear_name])))

    shoulder_tilt_deg = None
    if lm[LandmarkName.LEFT_SHOULDER].is_present and lm[LandmarkName.RIGHT_SHOULDER].is_present:
        shoulder_tilt_deg = angle_from_horizontal(
            vector(to_point(lm[LandmarkName.LEFT_SHOULDER]), to_point(lm[LandmarkName.RIGHT_SHOULDER]))
        )

    hip_tilt_deg = None
    if lm[LandmarkName.LEFT_HIP].is_present and lm[LandmarkName.RIGHT_HIP].is_present:
        hip_tilt_deg = angle_from_horizontal(
            vector(to_point(lm[LandmarkName.LEFT_HIP]), to_point(lm[LandmarkName.RIGHT_HIP]))
        )

    trunk_inclination_deg = None
    if all(
        lm[name].is_present
        for name in (
            LandmarkName.LEFT_SHOULDER,
            LandmarkName.RIGHT_SHOULDER,
            LandmarkName.LEFT_HIP,
            LandmarkName.RIGHT_HIP,
        )
    ):
        mid_shoulder = midpoint(to_point(lm[LandmarkName.LEFT_SHOULDER]), to_point(lm[LandmarkName.RIGHT_SHOULDER]))
        mid_hip = midpoint(to_point(lm[LandmarkName.LEFT_HIP]), to_point(lm[LandmarkName.RIGHT_HIP]))
        trunk_inclination_deg = angle_from_vertical(vector(mid_hip, mid_shoulder))

    body_inclination_deg = None
    if all(
        lm[name].is_present
        for name in (
            LandmarkName.LEFT_SHOULDER,
            LandmarkName.RIGHT_SHOULDER,
            LandmarkName.LEFT_ANKLE,
            LandmarkName.RIGHT_ANKLE,
        )
    ):
        mid_shoulder = midpoint(to_point(lm[LandmarkName.LEFT_SHOULDER]), to_point(lm[LandmarkName.RIGHT_SHOULDER]))
        mid_ankle = midpoint(to_point(lm[LandmarkName.LEFT_ANKLE]), to_point(lm[LandmarkName.RIGHT_ANKLE]))
        body_inclination_deg = angle_from_vertical(vector(mid_ankle, mid_shoulder))

    return PostureFeatures(
        forward_head_angle_deg=forward_head_angle_deg,
        shoulder_tilt_deg=shoulder_tilt_deg,
        hip_tilt_deg=hip_tilt_deg,
        trunk_inclination_deg=trunk_inclination_deg,
        body_inclination_deg=body_inclination_deg,
        left_knee_deg=joint_angles.left_knee_deg,
        right_knee_deg=joint_angles.right_knee_deg,
        joint_angles=joint_angles,
    )
