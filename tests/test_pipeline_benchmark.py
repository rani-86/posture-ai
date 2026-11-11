from src.evaluation.pipeline_benchmark import benchmark_pipeline
from src.utils.config import load_config


def test_benchmark_runs_on_synthetic_video(synthetic_video):
    config = load_config()
    result = benchmark_pipeline(synthetic_video["path"], config)

    assert result.num_frames > 0
    assert result.wall_clock_sec > 0
    assert result.fps > 0
    assert result.peak_memory_mb >= 0
