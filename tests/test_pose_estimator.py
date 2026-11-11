"""Unit tests for confidence filtering / missing-detection handling in PoseEstimator.

Rather than relying on MediaPipe actually detecting a person in a synthetic test image
(it won't, without a real photo), these tests construct a real PoseEstimator (so the
pinned mediapipe.solutions.pose API is exercised) but monkeypatch the underlying
`.process()` call with a fake MediaPipe-shaped result, isolating the confidence-filtering
logic this project added on top of MediaPipe.
"""
from types import SimpleNamespace

import numpy as np

from src.pose_estimation.landmark_schema import MEDIAPIPE_INDEX, LandmarkName
from src.pose_estimation.pose_estimator import PoseEstimator


def _fake_result(visible_indices: set[int]):
    landmark_list = []
    for i in range(33):
        visibility = 0.95 if i in visible_indices else 0.1
        landmark_list.append(SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=visibility))
    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=landmark_list))


def test_low_confidence_landmarks_are_filtered_out():
    estimator = PoseEstimator(landmark_confidence_threshold=0.5)
    visible = {MEDIAPIPE_INDEX[LandmarkName.NOSE]}  # only nose is confidently visible
    estimator._pose.process = lambda image_rgb: _fake_result(visible)

    pose_frame = estimator.estimate(np.zeros((48, 64, 3), dtype=np.uint8), frame_index=0, timestamp_sec=0.0)

    assert pose_frame.detected
    assert pose_frame.landmarks[LandmarkName.NOSE].is_present
    assert not pose_frame.landmarks[LandmarkName.LEFT_SHOULDER].is_present


def test_no_person_detected_marks_frame_undetected():
    estimator = PoseEstimator()
    estimator._pose.process = lambda image_rgb: SimpleNamespace(pose_landmarks=None)

    pose_frame = estimator.estimate(np.zeros((48, 64, 3), dtype=np.uint8), frame_index=5, timestamp_sec=0.5)

    assert not pose_frame.detected
    assert all(not lm.is_present for lm in pose_frame.landmarks.values())
    assert pose_frame.frame_index == 5
    assert pose_frame.timestamp_sec == 0.5
