"""Measures the Phase 2 pose-estimation pipeline's own throughput (FPS, wall-clock time,
peak memory) - separate from model accuracy, this is "how fast/heavy is the system
itself," answering the master prompt's video-processing evaluation requirement (#11).

Run against a synthetic video (see docs/phase6_notes.md for why): this measures the real
compute cost of frame decode + MediaPipe inference + smoothing + normalization + overlay
drawing, independent of whether a person is actually detected in frame - the per-frame
work is the same either way, so a synthetic video is a legitimate choice for a throughput
benchmark even though it can't validate detection accuracy (that needs a real video).
"""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

from src.pose_estimation.pipeline import run_pose_pipeline
from src.utils.config import PostureAIConfig


@dataclass
class BenchmarkResult:
    num_frames: int
    wall_clock_sec: float
    fps: float
    peak_memory_mb: float


def benchmark_pipeline(video_path: str | Path, config: PostureAIConfig) -> BenchmarkResult:
    tracemalloc.start()
    start = time.perf_counter()

    num_frames = 0
    for _ in run_pose_pipeline(video_path, config):
        num_frames += 1

    elapsed = time.perf_counter() - start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        num_frames=num_frames,
        wall_clock_sec=elapsed,
        fps=num_frames / elapsed if elapsed > 0 else float("inf"),
        peak_memory_mb=peak_bytes / (1024 * 1024),
    )
