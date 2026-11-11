"""Tests the dashboard's data-transformation layer (not the Streamlit widgets
themselves - UI rendering isn't unit-testable without a browser driver; see
docs/phase8_notes.md for how the UI was verified instead)."""
import tempfile
from pathlib import Path

import cv2
import numpy as np

from app.dashboard import _frame_records_df
from src.inference.report_pipeline import analyze_video
from src.utils.config import load_config


def _make_synthetic_video(n_frames: int = 12) -> Path:
    tmp_dir = Path(tempfile.mkdtemp())
    path = tmp_dir / "test.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(n_frames):
        writer.write(np.full((48, 64, 3), i % 256, dtype=np.uint8))
    writer.release()
    return path


def test_frame_records_df_has_expected_columns_and_length():
    video_path = _make_synthetic_video(n_frames=12)
    report = analyze_video(video_path, load_config())

    df = _frame_records_df(report)
    assert len(df) == 12
    assert set(df.columns) == {
        "frame_index",
        "timestamp_sec",
        "pose_detected",
        "trunk_inclination_deg",
        "left_knee_deg",
        "right_knee_deg",
        "forward_head_angle_deg",
        "shoulder_tilt_deg",
        "any_deviation",
    }
    # synthetic video has no real person, so pose was never detected
    assert not df["pose_detected"].any()
