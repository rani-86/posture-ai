"""The project's one explicit end-to-end test: a short sample video, run through the
entire pipeline (video I/O -> pose estimation -> smoothing/normalization -> posture
features -> rule-based deviation assessment -> activity prediction -> annotated video),
using the actual public entrypoint (analyze_video) a real caller would use - not a
synthetic shortcut through internal functions.

This uses a synthetic (no real person) video, same as the rest of the suite, for the
reason explained throughout the project: no real labeled video was available in this
environment. What this test proves is that the full pipeline runs start-to-finish
without errors and produces a well-formed report + playable annotated video - not that
pose detection is accurate (that needs a real video with a person in it, still an open
item - see docs/phase2_notes.md and docs/phase5_notes.md).
"""
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.inference.report_pipeline import analyze_video
from src.utils.config import load_config


def test_full_pipeline_end_to_end():
    tmp_dir = Path(tempfile.mkdtemp())
    video_path = tmp_dir / "sample.mp4"
    annotated_path = tmp_dir / "sample_annotated.mp4"

    # A short (2 second) sample video - what the spec calls for.
    fps, n_frames, size = 10, 20, (128, 96)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), fill_value=(i * 7) % 256, dtype=np.uint8))
    writer.release()
    assert video_path.exists() and video_path.stat().st_size > 0

    config = load_config()
    report = analyze_video(video_path, config, annotated_output_path=annotated_path)

    # Pipeline ran to completion and produced a report covering every frame.
    assert report.posture_summary.total_frames == n_frames
    assert len(report.frame_records) == n_frames

    # Annotated video was actually written and is a non-trivial, playable-looking file.
    assert annotated_path.exists()
    assert annotated_path.stat().st_size > 100

    readback = cv2.VideoCapture(str(annotated_path))
    assert readback.isOpened()
    readback_frame_count = int(readback.get(cv2.CAP_PROP_FRAME_COUNT))
    readback.release()
    assert readback_frame_count == n_frames

    # Activity prediction either ran (and carries its disclaimer) or was cleanly
    # skipped with a warning - never silently missing with no explanation.
    if report.activity_prediction is None:
        assert any("activity" in w.lower() for w in report.warnings)
    else:
        assert report.activity_prediction.disclaimer
