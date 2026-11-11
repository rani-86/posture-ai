import math

from src.feature_extraction.joint_angles import compute_joint_angles
from src.pose_estimation.landmark_schema import Landmark, LandmarkName, PoseFrame


def _blank_frame() -> PoseFrame:
    return PoseFrame(
        frame_index=0,
        timestamp_sec=0.0,
        detected=True,
        landmarks={name: Landmark(None, None, None, 0.0) for name in LandmarkName},
    )


def test_knee_angle_right_angle_bend():
    frame = _blank_frame()
    # Hip above knee, ankle to the side of knee -> 90 degree bend at the knee.
    frame.landmarks[LandmarkName.LEFT_HIP] = Landmark(0.0, 0.0, 0.0, 0.9)
    frame.landmarks[LandmarkName.LEFT_KNEE] = Landmark(0.0, 1.0, 0.0, 0.9)
    frame.landmarks[LandmarkName.LEFT_ANKLE] = Landmark(1.0, 1.0, 0.0, 0.9)

    angles = compute_joint_angles(frame)
    assert math.isclose(angles.left_knee_deg, 90.0, abs_tol=1e-6)


def test_knee_angle_fully_extended():
    frame = _blank_frame()
    frame.landmarks[LandmarkName.LEFT_HIP] = Landmark(0.0, 0.0, 0.0, 0.9)
    frame.landmarks[LandmarkName.LEFT_KNEE] = Landmark(0.0, 1.0, 0.0, 0.9)
    frame.landmarks[LandmarkName.LEFT_ANKLE] = Landmark(0.0, 2.0, 0.0, 0.9)

    angles = compute_joint_angles(frame)
    assert math.isclose(angles.left_knee_deg, 180.0, abs_tol=1e-6)


def test_missing_landmarks_yield_none():
    frame = _blank_frame()  # nothing populated
    angles = compute_joint_angles(frame)
    assert angles.left_knee_deg is None
    assert angles.right_elbow_deg is None
    assert angles.left_hip_deg is None
