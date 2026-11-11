"""Loader for the UI-PRMD dataset (University of Idaho Physical Rehabilitation Movement
Data), Kinect joint-angle, segmented-by-repetition release.

Ground truth available in this release: each file encodes a MOVEMENT TYPE label (m01-m10)
via its filename, performed by one of 10 healthy subjects (s01-s10), repeated 10 times
(e01-e10). This is a genuine, real label - the filename-to-movement mapping is the
dataset's own directory structure, not an assumption we are adding.

IMPORTANT CORRECTION vs. the Phase 1 plan: the base UI-PRMD release provided here does
NOT include a correct/incorrect movement-quality label - only movement type. The
originally planned "posture-quality classification" task (correct vs. deviant repetition)
is therefore not supported by this data and is NOT implemented; what IS implemented, and
genuinely supported, is 10-class MOVEMENT TYPE classification from joint-angle sequences.
See docs/phase4_notes.md for the full correction and reasoning.

Each file is a CSV of floats, one row per frame, 66 columns (22 Kinect skeleton joints x
3 Euler rotation angles per joint). The exact column-to-joint-name mapping is documented
in UI-PRMD's own PDF documentation (not redistributed here); rather than hard-code an
unverified mapping, this loader keeps the 66 channels in their original file order and
uses them directly as model input - correct for classification (the channels are a
consistent, real feature space), but it means per-joint-name feature importance
(e.g. "which joint drives this prediction") is only available as "channel index N", not
a named joint, unless that mapping is added later from the official documentation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

NUM_MOVEMENT_CLASSES = 10
NUM_CHANNELS = 66

_FILENAME_PATTERN = re.compile(r"m(?P<movement>\d{2})_s(?P<subject>\d{2})_e(?P<episode>\d{2})_angles\.txt")


class UIPRMDLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class Episode:
    movement_id: int  # 1-10, the class label
    subject_id: int  # 1-10
    episode_id: int  # 1-10 (repetition number)
    sequence: np.ndarray  # shape (T, 66), float32


def _parse_filename(path: Path) -> tuple[int, int, int]:
    match = _FILENAME_PATTERN.match(path.name)
    if not match:
        raise UIPRMDLoadError(f"Unrecognized UI-PRMD filename format: {path.name}")
    return int(match["movement"]), int(match["subject"]), int(match["episode"])


def load_episode(path: str | Path) -> Episode:
    path = Path(path)
    movement_id, subject_id, episode_id = _parse_filename(path)
    sequence = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if sequence.ndim == 1:  # a single-frame file would collapse to 1D
        sequence = sequence.reshape(1, -1)
    if sequence.shape[1] != NUM_CHANNELS:
        raise UIPRMDLoadError(
            f"{path.name}: expected {NUM_CHANNELS} channels, got {sequence.shape[1]}"
        )
    return Episode(movement_id=movement_id, subject_id=subject_id, episode_id=episode_id, sequence=sequence)


def load_all_episodes(angles_dir: str | Path) -> list[Episode]:
    angles_dir = Path(angles_dir)
    paths = sorted(angles_dir.glob("*_angles.txt"))
    if not paths:
        raise UIPRMDLoadError(f"No *_angles.txt files found in {angles_dir}")

    episodes = [load_episode(p) for p in paths]
    logger.info(
        f"Loaded {len(episodes)} UI-PRMD episodes from {angles_dir} "
        f"({len({e.movement_id for e in episodes})} movement classes, "
        f"{len({e.subject_id for e in episodes})} subjects)"
    )
    return episodes


def subject_wise_split(
    episodes: list[Episode], train_subjects: set[int], val_subjects: set[int], test_subjects: set[int]
) -> tuple[list[Episode], list[Episode], list[Episode]]:
    """Splits by subject, not by episode - repetitions from the same subject are
    correlated, so mixing a subject's episodes across train/val/test would leak
    subject-specific movement style into the evaluation sets."""
    overlap = (train_subjects & val_subjects) | (train_subjects & test_subjects) | (val_subjects & test_subjects)
    if overlap:
        raise ValueError(f"Subject sets must be disjoint, overlap: {overlap}")

    train = [e for e in episodes if e.subject_id in train_subjects]
    val = [e for e in episodes if e.subject_id in val_subjects]
    test = [e for e in episodes if e.subject_id in test_subjects]
    return train, val, test
