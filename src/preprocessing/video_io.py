"""Video reading: metadata extraction and frame-by-frame iteration with resampling.

Frames are yielded one at a time (a generator, not a list) so that long videos are never
fully loaded into memory - callers process and discard each frame before the next is read.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VideoReadError(RuntimeError):
    """Raised when a video file cannot be opened or decoded."""


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


@dataclass(frozen=True)
class Frame:
    """One decoded frame plus its position in the original (un-resampled) video."""

    index: int
    timestamp_sec: float
    image: np.ndarray  # BGR, as returned by OpenCV


def get_video_metadata(path: str | Path) -> VideoMetadata:
    path = Path(path)
    if not path.exists():
        raise VideoReadError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoReadError(f"OpenCV could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    if fps <= 0:
        raise VideoReadError(f"Video reports invalid FPS ({fps}): {path}")

    return VideoMetadata(path=path, fps=fps, frame_count=frame_count, width=width, height=height)


def iter_frames(path: str | Path, target_fps: int | None = None) -> Iterator[Frame]:
    """Yield frames from `path`, optionally downsampled to `target_fps`.

    Downsampling is done by skipping source frames (not by re-encoding), so the original
    timestamp of each yielded frame is preserved exactly (`index / source_fps`), rather
    than being recomputed at the target rate - callers that need absolute time alignment
    with the source video get it, even though frames are sparser.
    """
    metadata = get_video_metadata(path)
    source_fps = metadata.fps

    if target_fps is not None and target_fps > 0 and target_fps < source_fps:
        keep_every = round(source_fps / target_fps)
    else:
        keep_every = 1

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoReadError(f"OpenCV could not open video: {path}")

    logger.info(
        f"Reading {metadata.path.name}: {metadata.frame_count} frames @ {source_fps:.2f}fps "
        f"(keeping 1 of every {keep_every} frames)"
    )

    index = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if index % keep_every == 0:
                yield Frame(index=index, timestamp_sec=index / source_fps, image=image)
            index += 1
    finally:
        cap.release()


class AnnotatedVideoWriter:
    """Incrementally writes annotated frames to an output video (no buffering in memory)."""

    def __init__(self, output_path: str | Path, fps: float, width: int, height: int):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise VideoReadError(f"Could not open video writer for: {output_path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "AnnotatedVideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
