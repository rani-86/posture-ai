"""Orchestrates the full per-video analysis: pose estimation (Phase 2) -> posture
features + rule-based deviation flags (Phase 3) -> activity prediction (Phase 5 model,
on real pipeline output) -> a structured report + annotated video.

This is the one place that combines genuinely-validated components (pose estimation,
posture analysis) with a component trained only on synthetic data (activity
recognition) - the report's `activity_prediction.disclaimer` field exists so no caller
can present that prediction without the caveat attached.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.data.activity_dataset import ACTIVITY_CLASSES
from src.feature_extraction.pose_channels import pose_frame_to_channels
from src.inference.model_registry import get_activity_model
from src.pose_estimation.pipeline import run_pose_pipeline
from src.posture_analysis.posture_engine import assess_posture
from src.posture_analysis.posture_features import compute_posture_features
from src.utils.config import PostureAIConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

ACTIVITY_MODEL_DISCLAIMER = (
    "This activity model was trained only on synthetic, procedurally-generated "
    "placeholder data (see docs/phase5_notes.md) - no real labeled video clips were "
    "available. Treat this prediction as a pipeline demonstration, not a validated "
    "real-world classification."
)


@dataclass
class PostureSummary:
    frames_with_pose_detected: int
    total_frames: int
    forward_head_deviation_rate: float | None
    shoulder_asymmetry_deviation_rate: float | None
    trunk_inclination_deviation_rate: float | None
    mean_trunk_inclination_deg: float | None


@dataclass
class ActivityPrediction:
    predicted_label: str
    confidence: float
    class_probabilities: dict[str, float]
    disclaimer: str = ACTIVITY_MODEL_DISCLAIMER


@dataclass
class FrameRecord:
    """Per-frame data for timeline/joint-angle charts - the Streamlit dashboard (Phase 8)
    is the first consumer; the FastAPI report (Phase 7) intentionally stays aggregate-only
    to keep that response small, so this lives on the report object rather than the API
    schema."""

    frame_index: int
    timestamp_sec: float
    pose_detected: bool
    trunk_inclination_deg: float | None
    left_knee_deg: float | None
    right_knee_deg: float | None
    forward_head_angle_deg: float | None
    shoulder_tilt_deg: float | None
    any_deviation: bool | None


@dataclass
class VideoAnalysisReport:
    video_path: str
    annotated_video_path: str | None
    posture_summary: PostureSummary
    activity_prediction: ActivityPrediction | None
    frame_records: list[FrameRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _rate_or_none(flags: list[bool]) -> float | None:
    return float(np.mean(flags)) if flags else None


def analyze_video(
    video_path: str | Path, config: PostureAIConfig, annotated_output_path: str | Path | None = None
) -> VideoAnalysisReport:
    warnings: list[str] = []

    detected_count = 0
    total_count = 0
    forward_head_flags: list[bool] = []
    shoulder_flags: list[bool] = []
    trunk_flags: list[bool] = []
    trunk_angles: list[float] = []
    channel_sequence: list[np.ndarray] = []
    frame_records: list[FrameRecord] = []

    for normalized_pose, raw_pose in run_pose_pipeline(video_path, config, annotated_output_path):
        total_count += 1
        detected_count += int(raw_pose.detected)

        features = compute_posture_features(normalized_pose)
        deviation = assess_posture(features, config.posture_thresholds)

        if deviation.forward_head_deviation is not None:
            forward_head_flags.append(deviation.forward_head_deviation)
        if deviation.shoulder_asymmetry_deviation is not None:
            shoulder_flags.append(deviation.shoulder_asymmetry_deviation)
        if deviation.trunk_inclination_deviation is not None:
            trunk_flags.append(deviation.trunk_inclination_deviation)
        if features.trunk_inclination_deg is not None:
            trunk_angles.append(features.trunk_inclination_deg)

        frame_records.append(
            FrameRecord(
                frame_index=normalized_pose.frame_index,
                timestamp_sec=normalized_pose.timestamp_sec,
                pose_detected=raw_pose.detected,
                trunk_inclination_deg=features.trunk_inclination_deg,
                left_knee_deg=features.left_knee_deg,
                right_knee_deg=features.right_knee_deg,
                forward_head_angle_deg=features.forward_head_angle_deg,
                shoulder_tilt_deg=features.shoulder_tilt_deg,
                any_deviation=deviation.any_deviation,
            )
        )

        channel_sequence.append(pose_frame_to_channels(normalized_pose))

    if total_count == 0:
        raise ValueError(f"No frames could be read from video: {video_path}")

    detection_rate = detected_count / total_count
    if detection_rate < 0.5:
        warnings.append(
            f"Pose detected in only {detection_rate:.0%} of frames - posture/activity "
            "results below may be unreliable for most of this video."
        )

    posture_summary = PostureSummary(
        frames_with_pose_detected=detected_count,
        total_frames=total_count,
        forward_head_deviation_rate=_rate_or_none(forward_head_flags),
        shoulder_asymmetry_deviation_rate=_rate_or_none(shoulder_flags),
        trunk_inclination_deviation_rate=_rate_or_none(trunk_flags),
        mean_trunk_inclination_deg=_mean_or_none(trunk_angles),
    )

    activity_prediction = None
    try:
        bundle = get_activity_model()
        sequence = np.stack(channel_sequence)
        predicted_idx, probs = bundle.predict(sequence)
        activity_prediction = ActivityPrediction(
            predicted_label=ACTIVITY_CLASSES[predicted_idx],
            confidence=float(probs[predicted_idx]),
            class_probabilities={ACTIVITY_CLASSES[i]: float(p) for i, p in enumerate(probs)},
        )
    except FileNotFoundError as e:
        logger.warning(f"Activity model unavailable: {e}")
        warnings.append("Activity prediction unavailable: model checkpoint not found.")

    return VideoAnalysisReport(
        video_path=str(video_path),
        annotated_video_path=str(annotated_output_path) if annotated_output_path else None,
        posture_summary=posture_summary,
        activity_prediction=activity_prediction,
        frame_records=frame_records,
        warnings=warnings,
    )
