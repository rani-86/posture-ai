"""Two feature representations of a UI-PRMD episode, for the two model families compared
in Phase 4:

- `summary_statistics_features`: a fixed-length vector (mean/std/min/max/range per
  channel) for XGBoost, which needs tabular, non-sequential input.
- `pad_or_truncate_sequence`: a fixed-length (T, C) sequence for the LSTM/GRU, which
  consumes the raw per-frame signal directly rather than summarizing it away.
"""
from __future__ import annotations

import numpy as np

from src.data.uiprmd_dataset import Episode


def summary_statistics_features(episode: Episode) -> np.ndarray:
    seq = episode.sequence  # (T, C)
    mean = seq.mean(axis=0)
    std = seq.std(axis=0)
    minimum = seq.min(axis=0)
    maximum = seq.max(axis=0)
    value_range = maximum - minimum
    return np.concatenate([mean, std, minimum, maximum, value_range]).astype(np.float32)


def pad_or_truncate_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Pads with the last frame repeated, or truncates from the end, to `target_length`.

    Repeating the last frame (rather than zero-padding) avoids injecting an artificial
    "snap to origin" motion at the end of shorter sequences, which would look like a
    sudden movement to a sequence model.
    """
    t = sequence.shape[0]
    if t == target_length:
        return sequence
    if t > target_length:
        return sequence[:target_length]
    pad = np.repeat(sequence[-1:], target_length - t, axis=0)
    return np.concatenate([sequence, pad], axis=0)
