"""torch Dataset wrapping labeled sequence samples as fixed-length padded sequences.

Shared by UI-PRMD movement classification (Phase 4) and self-curated activity
recognition (Phase 5) - any object with a `.sequence` (T, C) array works, as long as a
`label_fn` is given to extract its 0-indexed class label.
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.uiprmd_dataset import Episode
from src.feature_extraction.sequence_features import pad_or_truncate_sequence


class _HasSequence(Protocol):
    sequence: np.ndarray


def uiprmd_label_fn(episode: Episode) -> int:
    return episode.movement_id - 1  # 0-indexed for CrossEntropyLoss


class PoseSequenceDataset(Dataset):
    def __init__(
        self,
        samples: list[_HasSequence],
        target_length: int,
        normalize_stats: tuple[np.ndarray, np.ndarray] | None = None,
        label_fn: Callable[[_HasSequence], int] = uiprmd_label_fn,
    ):
        self.samples = samples
        self.target_length = target_length
        self.label_fn = label_fn
        # Per-channel z-score normalization - fit on train split only, passed in for val/test
        # so no evaluation-set statistics leak into feature scaling.
        if normalize_stats is None:
            all_frames = np.concatenate([s.sequence for s in samples], axis=0)
            self.mean = all_frames.mean(axis=0)
            self.std = all_frames.std(axis=0) + 1e-6
        else:
            self.mean, self.std = normalize_stats

    @property
    def normalize_stats(self) -> tuple[np.ndarray, np.ndarray]:
        return self.mean, self.std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        seq = pad_or_truncate_sequence(sample.sequence, self.target_length)
        seq = (seq - self.mean) / self.std
        label = self.label_fn(sample)
        return torch.from_numpy(seq.astype(np.float32)), torch.tensor(label, dtype=torch.long)
