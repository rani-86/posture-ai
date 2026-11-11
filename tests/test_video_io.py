import pytest

from src.preprocessing.video_io import VideoReadError, get_video_metadata, iter_frames


def test_metadata_matches_known_video(synthetic_video):
    meta = get_video_metadata(synthetic_video["path"])
    assert meta.frame_count == synthetic_video["n_frames"]
    assert meta.width == synthetic_video["width"]
    assert meta.height == synthetic_video["height"]
    assert abs(meta.fps - synthetic_video["fps"]) < 0.5


def test_metadata_missing_file_raises(tmp_path):
    with pytest.raises(VideoReadError):
        get_video_metadata(tmp_path / "does_not_exist.mp4")


def test_iter_frames_full_rate_preserves_timestamps(synthetic_video):
    frames = list(iter_frames(synthetic_video["path"], target_fps=None))
    assert len(frames) == synthetic_video["n_frames"]

    timestamps = [f.timestamp_sec for f in frames]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == 0.0
    expected_last = (synthetic_video["n_frames"] - 1) / synthetic_video["fps"]
    assert abs(timestamps[-1] - expected_last) < 1e-6


def test_iter_frames_downsamples_to_target_fps(synthetic_video):
    target_fps = 10  # source is 30fps -> keep every 3rd frame
    frames = list(iter_frames(synthetic_video["path"], target_fps=target_fps))

    expected_count = synthetic_video["n_frames"] // 3
    assert len(frames) == expected_count

    # Original timestamps (not recomputed at target rate) must be preserved.
    for f in frames:
        assert f.index % 3 == 0
        assert abs(f.timestamp_sec - f.index / synthetic_video["fps"]) < 1e-9


def test_iter_frames_target_fps_above_source_keeps_every_frame(synthetic_video):
    frames = list(iter_frames(synthetic_video["path"], target_fps=1000))
    assert len(frames) == synthetic_video["n_frames"]
