import numpy as np

from src.pose_estimation.landmark_schema import Landmark, LandmarkName, PoseFrame
from src.visualization.skeleton import draw_skeleton


def test_draw_skeleton_does_not_mutate_input_and_preserves_shape():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    landmarks = {name: Landmark(None, None, None, 0.0) for name in LandmarkName}
    landmarks[LandmarkName.LEFT_SHOULDER] = Landmark(0.3, 0.3, 0.0, 0.9)
    landmarks[LandmarkName.RIGHT_SHOULDER] = Landmark(0.7, 0.3, 0.0, 0.9)
    frame = PoseFrame(frame_index=0, timestamp_sec=0.0, detected=True, landmarks=landmarks)

    original = image.copy()
    annotated = draw_skeleton(image, frame)

    assert np.array_equal(image, original)  # input untouched
    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)  # something was actually drawn


def test_draw_skeleton_handles_fully_missing_frame():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    landmarks = {name: Landmark(None, None, None, 0.0) for name in LandmarkName}
    frame = PoseFrame(frame_index=0, timestamp_sec=0.0, detected=False, landmarks=landmarks)

    annotated = draw_skeleton(image, frame)
    assert np.array_equal(annotated, image)  # nothing to draw, no crash
