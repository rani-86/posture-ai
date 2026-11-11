"""Tests src/inference/report_pipeline.analyze_video directly (report generation),
separate from the API (test_api.py) and dashboard (test_dashboard.py) layers that
consume it - this is the one place that checks the report's own structure and content
guarantees regardless of which caller is using it."""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.inference.report_pipeline import ActivityPrediction, FrameRecord, PostureSummary, analyze_video
from src.preprocessing.video_io import VideoReadError
from src.utils.config import load_config


def _make_synthetic_video(n_frames: int, tmp_path: Path, fps: int = 10, size: tuple[int, int] = (64, 48)) -> Path:
    path = tmp_path / "video.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), fill_value=i % 256, dtype=np.uint8))
    writer.release()
    return path


def test_report_structure_and_types(tmp_path):
    video_path = _make_synthetic_video(15, tmp_path)
    report = analyze_video(video_path, load_config())

    assert isinstance(report.posture_summary, PostureSummary)
    assert report.posture_summary.total_frames == 15
    assert report.posture_summary.frames_with_pose_detected == 0  # no real person in frame

    assert len(report.frame_records) == 15
    assert all(isinstance(r, FrameRecord) for r in report.frame_records)
    assert [r.frame_index for r in report.frame_records] == list(range(15))

    # no pose detected anywhere -> posture features are all None, not fabricated zeros
    assert all(r.trunk_inclination_deg is None for r in report.frame_records)

    assert report.activity_prediction is None or isinstance(report.activity_prediction, ActivityPrediction)
    assert any("only 0%" in w for w in report.warnings)


def test_report_writes_annotated_video_when_requested(tmp_path):
    video_path = _make_synthetic_video(10, tmp_path)
    annotated_path = tmp_path / "annotated.mp4"

    report = analyze_video(video_path, load_config(), annotated_output_path=annotated_path)

    assert annotated_path.exists()
    assert annotated_path.stat().st_size > 0
    assert report.annotated_video_path == str(annotated_path)


def test_report_raises_on_empty_video(tmp_path):
    path = tmp_path / "empty.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    writer.release()  # zero frames written -> OpenCV can't even open the resulting file

    with pytest.raises(VideoReadError):
        analyze_video(path, load_config())


def test_activity_prediction_carries_disclaimer_when_present(tmp_path):
    video_path = _make_synthetic_video(20, tmp_path)
    report = analyze_video(video_path, load_config())

    if report.activity_prediction is not None:
        assert "synthetic" in report.activity_prediction.disclaimer.lower()
        assert len(report.activity_prediction.class_probabilities) == 5
        assert abs(sum(report.activity_prediction.class_probabilities.values()) - 1.0) < 1e-4
