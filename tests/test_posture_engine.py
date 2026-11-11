from src.feature_extraction.joint_angles import JointAngles
from src.posture_analysis.posture_engine import assess_posture
from src.posture_analysis.posture_features import PostureFeatures
from src.utils.config import PostureThresholds

THRESHOLDS = PostureThresholds(
    forward_head_angle_deg=50,
    shoulder_asymmetry_deg=5,
    trunk_inclination_deg=10,
    knee_angle_deep_bend_deg=90,
)

EMPTY_JOINT_ANGLES = JointAngles(None, None, None, None, None, None)


def _features(**overrides) -> PostureFeatures:
    base = dict(
        forward_head_angle_deg=60.0,
        shoulder_tilt_deg=1.0,
        hip_tilt_deg=1.0,
        trunk_inclination_deg=2.0,
        body_inclination_deg=2.0,
        left_knee_deg=175.0,
        right_knee_deg=175.0,
        joint_angles=EMPTY_JOINT_ANGLES,
    )
    base.update(overrides)
    return PostureFeatures(**base)


def test_normal_posture_flags_no_deviation():
    result = assess_posture(_features(), THRESHOLDS)
    assert result.forward_head_deviation is False
    assert result.shoulder_asymmetry_deviation is False
    assert result.trunk_inclination_deviation is False
    assert result.any_deviation is False


def test_forward_head_below_threshold_flags_deviation():
    result = assess_posture(_features(forward_head_angle_deg=40.0), THRESHOLDS)
    assert result.forward_head_deviation is True
    assert result.any_deviation is True


def test_shoulder_asymmetry_above_threshold_flags_deviation():
    result = assess_posture(_features(shoulder_tilt_deg=8.0), THRESHOLDS)
    assert result.shoulder_asymmetry_deviation is True
    assert result.any_deviation is True


def test_deep_knee_bend_is_informational_not_in_any_deviation():
    result = assess_posture(_features(left_knee_deg=70.0, right_knee_deg=70.0), THRESHOLDS)
    assert result.deep_knee_bend is True
    # a deep knee bend alone (e.g. intentional squat) should not flag "any_deviation"
    assert result.any_deviation is False


def test_missing_features_yield_undetermined_not_false():
    result = assess_posture(_features(forward_head_angle_deg=None), THRESHOLDS)
    assert result.forward_head_deviation is None
    # other determined flags still contribute to any_deviation
    assert result.any_deviation is False
