from src.pose_estimation.landmark_schema import Landmark, LandmarkName, PoseFrame
from src.pose_estimation.normalization import normalize_pose_frame


def _frame_with(points: dict[LandmarkName, tuple[float, float]]) -> PoseFrame:
    landmarks = {name: Landmark(None, None, None, 0.0) for name in LandmarkName}
    for name, (x, y) in points.items():
        landmarks[name] = Landmark(x, y, 0.0, visibility=0.9)
    return PoseFrame(frame_index=0, timestamp_sec=0.0, detected=True, landmarks=landmarks)


def test_normalization_centers_on_mid_hip():
    frame = _frame_with(
        {
            LandmarkName.LEFT_HIP: (0.4, 0.6),
            LandmarkName.RIGHT_HIP: (0.6, 0.6),
            LandmarkName.LEFT_SHOULDER: (0.4, 0.2),
            LandmarkName.RIGHT_SHOULDER: (0.6, 0.2),
        }
    )
    out = normalize_pose_frame(frame)

    # mid-hip is (0.5, 0.6); mid-shoulder is (0.5, 0.2) -> torso length 0.4, directly above hip
    left_hip = out.landmarks[LandmarkName.LEFT_HIP]
    right_hip = out.landmarks[LandmarkName.RIGHT_HIP]
    assert abs((left_hip.x + right_hip.x) / 2) < 1e-9
    assert abs((left_hip.y + right_hip.y) / 2) < 1e-9

    mid_shoulder_y = (
        out.landmarks[LandmarkName.LEFT_SHOULDER].y + out.landmarks[LandmarkName.RIGHT_SHOULDER].y
    ) / 2
    assert abs(abs(mid_shoulder_y) - 1.0) < 1e-9  # torso length rescaled to 1.0


def test_normalization_passthrough_when_reference_points_missing():
    frame = _frame_with({LandmarkName.LEFT_HIP: (0.4, 0.6)})  # missing right hip + shoulders
    out = normalize_pose_frame(frame)
    assert out.landmarks[LandmarkName.LEFT_HIP].x == 0.4
    assert out.landmarks[LandmarkName.LEFT_HIP].y == 0.6
