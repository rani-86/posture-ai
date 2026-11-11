"""torch Dataset wrapping UI-PRMD episodes as fixed-length padded sequences."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.uiprmd_dataset import Episode
from src.feature_extraction.sequence_features import pad_or_truncate_sequence


class PoseSequenceDataset(Dataset):
    def __init__(self, episodes: list[Episode], target_length: int, normalize_stats: tuple[np.ndarray, np.ndarray] | None = None):
        self.episodes = episodes
        self.target_length = target_length
        # Per-channel z-score normalization - fit on train split only, passed in for val/test
        # so no evaluation-set statistics leak into feature scaling.
        if normalize_stats is None:
            all_frames = np.concatenate([e.sequence for e in episodes], axis=0)
            self.mean = all_frames.mean(axis=0)
            self.std = all_frames.std(axis=0) + 1e-6
        else:
            self.mean, self.std = normalize_stats

    @property
    def normalize_stats(self) -> tuple[np.ndarray, np.ndarray]:
        return self.mean, self.std

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        episode = self.episodes[idx]
        seq = pad_or_truncate_sequence(episode.sequence, self.target_length)
        seq = (seq - self.mean) / self.std
        label = episode.movement_id - 1  # 0-indexed for CrossEntropyLoss
        return torch.from_numpy(seq.astype(np.float32)), torch.tensor(label, dtype=torch.long)
