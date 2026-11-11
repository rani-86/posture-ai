import math

from src.pose_estimation.landmark_schema import Landmark, LandmarkName, PoseFrame
from src.posture_analysis.posture_features import compute_posture_features


def _blank_frame() -> PoseFrame:
    return PoseFrame(
        frame_index=0,
        timestamp_sec=0.0,
        detected=True,
        landmarks={name: Landmark(None, None, None, 0.0) for name in LandmarkName},
    )


def test_level_shoulders_give_zero_tilt():
    frame = _blank_frame()
    frame.landmarks[LandmarkName.LEFT_SHOULDER] = Landmark(0.3, 0.3, 0.0, 0.9)
    frame.landmarks[LandmarkName.RIGHT_SHOULDER] = Landmark(0.7, 0.3, 0.0, 0.9)

    features = compute_posture_features(frame)
    assert math.isclose(features.shoulder_tilt_deg, 0.0, abs_tol=1e-6)


def test_tilted_shoulders_give_nonzero_tilt():
    frame = _blank_frame()
    frame.landmarks[LandmarkName.LEFT_SHOULDER] = Landmark(0.3, 0.2, 0.0, 0.9)
    frame.landmarks[LandmarkName.RIGHT_SHOULDER] = Landmark(0.7, 0.4, 0.0, 0.9)

    features = compute_posture_features(frame)
    assert features.shoulder_tilt_deg > 0


def test_upright_trunk_gives_zero_inclination():
    frame = _blank_frame()
    frame.landmarks[LandmarkName.LEFT_SHOULDER] = Landmark(0.4, 0.2, 0.0, 0.9)
    frame.landmarks[LandmarkName.RIGHT_SHOULDER] = Landmark(0.6, 0.2, 0.0, 0.9)
    frame.landmarks[LandmarkName.LEFT_HIP] = Landmark(0.4, 0.6, 0.0, 0.9)
    frame.landmarks[LandmarkName.RIGHT_HIP] = Landmark(0.6, 0.6, 0.0, 0.9)

    features = compute_posture_features(frame)
    assert math.isclose(features.trunk_inclination_deg, 0.0, abs_tol=1e-6)


def test_leaning_trunk_gives_nonzero_inclination():
    frame = _blank_frame()
    frame.landmarks[LandmarkName.LEFT_SHOULDER] = Landmark(0.6, 0.2, 0.0, 0.9)
    frame.landmarks[LandmarkName.RIGHT_SHOULDER] = Landmark(0.8, 0.2, 0.0, 0.9)
    frame.landmarks[LandmarkName.LEFT_HIP] = Landmark(0.4, 0.6, 0.0, 0.9)
    frame.landmarks[LandmarkName.RIGHT_HIP] = Landmark(0.6, 0.6, 0.0, 0.9)

    features = compute_posture_features(frame)
    assert features.trunk_inclination_deg > 0


def test_missing_reference_points_give_none():
    frame = _blank_frame()
    features = compute_posture_features(frame)
    assert features.shoulder_tilt_deg is None
    assert features.trunk_inclination_deg is None
    assert features.forward_head_angle_deg is None
    assert features.body_inclination_deg is None


def test_forward_head_angle_picks_best_visible_side():
    frame = _blank_frame()
    # Only the right ear/shoulder pair is present -> must use that side without error.
    frame.landmarks[LandmarkName.RIGHT_SHOULDER] = Landmark(0.5, 0.5, 0.0, 0.9)
    frame.landmarks[LandmarkName.RIGHT_EAR] = Landmark(0.6, 0.3, 0.0, 0.9)

    features = compute_posture_features(frame)
    assert features.forward_head_angle_deg is not None
    assert 0 <= features.forward_head_angle_deg <= 90
