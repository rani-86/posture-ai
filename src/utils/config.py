"""Loads configs/config.yaml into a typed, dot-accessible structure."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class VideoConfig:
    target_fps: int
    max_frames_in_memory: int


@dataclass(frozen=True)
class PoseEstimationConfig:
    backend: str
    min_detection_confidence: float
    min_tracking_confidence: float
    landmark_confidence_threshold: float
    smoothing_window: int


@dataclass(frozen=True)
class PostureThresholds:
    forward_head_angle_deg: float
    shoulder_asymmetry_deg: float
    trunk_inclination_deg: float
    knee_angle_deep_bend_deg: float


@dataclass(frozen=True)
class SequenceModelConfig:
    window_size: int
    stride: int
    hidden_size: int
    num_layers: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    val_split: float
    test_split: float


@dataclass(frozen=True)
class PostureAIConfig:
    project_root: Path
    seed: int
    device: str
    raw_data_dir: Path
    processed_data_dir: Path
    checkpoints_dir: Path
    reports_dir: Path
    video: VideoConfig
    pose_estimation: PoseEstimationConfig
    posture_thresholds: PostureThresholds
    sequence_model: SequenceModelConfig
    training: TrainingConfig
    raw: dict[str, Any] = field(repr=False)


def load_config(path: str | Path | None = None) -> PostureAIConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    root = config_path.resolve().parents[1]
    paths = raw["paths"]

    return PostureAIConfig(
        project_root=root,
        seed=raw["project"]["seed"],
        device=raw["project"]["device"],
        raw_data_dir=root / paths["raw_data_dir"],
        processed_data_dir=root / paths["processed_data_dir"],
        checkpoints_dir=root / paths["checkpoints_dir"],
        reports_dir=root / paths["reports_dir"],
        video=VideoConfig(**raw["video"]),
        pose_estimation=PoseEstimationConfig(**raw["pose_estimation"]),
        posture_thresholds=PostureThresholds(**raw["posture_thresholds"]),
        sequence_model=SequenceModelConfig(**raw["sequence_model"]),
        training=TrainingConfig(**raw["training"]),
        raw=raw,
    )
