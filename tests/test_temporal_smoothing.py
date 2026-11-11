from src.pose_estimation.landmark_schema import Landmark, LandmarkName, PoseFrame
from src.pose_estimation.temporal_smoothing import LandmarkSmoother


def _frame(frame_index: int, nose_xy: tuple[float, float] | None) -> PoseFrame:
    landmarks = {name: Landmark(None, None, None, 0.0) for name in LandmarkName}
    if nose_xy is not None:
        landmarks[LandmarkName.NOSE] = Landmark(nose_xy[0], nose_xy[1], 0.0, visibility=0.9)
    return PoseFrame(frame_index=frame_index, timestamp_sec=frame_index / 30, detected=True, landmarks=landmarks)


def test_smoothing_averages_within_window():
    smoother = LandmarkSmoother(window_size=3)
    smoother.smooth(_frame(0, (0.0, 0.0)))
    smoother.smooth(_frame(1, (0.3, 0.0)))
    out = smoother.smooth(_frame(2, (0.6, 0.0)))

    nose = out.landmarks[LandmarkName.NOSE]
    assert abs(nose.x - 0.3) < 1e-9  # mean of 0.0, 0.3, 0.6


def test_missing_landmark_holds_last_known_value():
    smoother = LandmarkSmoother(window_size=5)
    smoother.smooth(_frame(0, (0.5, 0.5)))
    out = smoother.smooth(_frame(1, nose_xy=None))  # detection dropped this frame

    nose = out.landmarks[LandmarkName.NOSE]
    assert nose.is_present
    assert nose.x == 0.5 and nose.y == 0.5
    assert nose.visibility == 0.0  # flags this value as held-forward, not observed


def test_landmark_never_seen_stays_absent():
    smoother = LandmarkSmoother(window_size=5)
    out = smoother.smooth(_frame(0, nose_xy=None))

    assert not out.landmarks[LandmarkName.NOSE].is_present
