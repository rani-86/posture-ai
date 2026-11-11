"""Phase 2 end-to-end pipeline: video in -> per-frame pose landmarks out (+ optional
annotated video), processing frame by frame without loading the whole video into memory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from src.pose_estimation.landmark_schema import PoseFrame
from src.pose_estimation.normalization import normalize_pose_frame
from src.pose_estimation.pose_estimator import PoseEstimator
from src.pose_estimation.temporal_smoothing import LandmarkSmoother
from src.preprocessing.video_io import AnnotatedVideoWriter, get_video_metadata, iter_frames
from src.utils.config import PostureAIConfig
from src.utils.logging_utils import get_logger
from src.visualization.skeleton import draw_skeleton

logger = get_logger(__name__)


def run_pose_pipeline(
    video_path: str | Path,
    config: PostureAIConfig,
    annotated_output_path: str | Path | None = None,
) -> Iterator[tuple[PoseFrame, PoseFrame]]:
    """Yields (smoothed_and_normalized_frame, raw_frame) pairs in temporal order.

    If `annotated_output_path` is given, writes a skeleton-overlay video alongside
    (drawn from the smoothed, non-normalized landmarks, since normalized coordinates are
    not in image space).
    """
    metadata = get_video_metadata(video_path)
    smoother = LandmarkSmoother(window_size=config.pose_estimation.smoothing_window)

    writer = None
    if annotated_output_path is not None:
        output_fps = min(metadata.fps, config.video.target_fps)
        writer = AnnotatedVideoWriter(
            annotated_output_path, fps=output_fps, width=metadata.width, height=metadata.height
        )

    try:
        with PoseEstimator(
            min_detection_confidence=config.pose_estimation.min_detection_confidence,
            min_tracking_confidence=config.pose_estimation.min_tracking_confidence,
            landmark_confidence_threshold=config.pose_estimation.landmark_confidence_threshold,
        ) as estimator:
            detected_count = 0
            total_count = 0
            for frame in iter_frames(video_path, target_fps=config.video.target_fps):
                raw_pose = estimator.estimate(frame.image, frame.index, frame.timestamp_sec)
                smoothed_pose = smoother.smooth(raw_pose)
                normalized_pose = normalize_pose_frame(smoothed_pose)

                total_count += 1
                detected_count += int(raw_pose.detected)

                if writer is not None:
                    writer.write(draw_skeleton(frame.image, smoothed_pose))

                yield normalized_pose, raw_pose

            if total_count > 0:
                logger.info(
                    f"Pose detected in {detected_count}/{total_count} processed frames "
                    f"({detected_count / total_count:.0%})"
                )
    finally:
        if writer is not None:
            writer.close()
