from src.utils.config import load_config


def test_load_default_config():
    cfg = load_config()
    assert cfg.video.target_fps > 0
    assert 0 < cfg.pose_estimation.landmark_confidence_threshold <= 1
    assert cfg.sequence_model.window_size > 0
    assert cfg.checkpoints_dir.name == "checkpoints"
