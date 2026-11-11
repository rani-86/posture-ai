"""Error analysis helpers: going beyond a single accuracy number to find WHICH classes
get confused and whether that confusion is model-specific or structural to the data."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConfusedPair:
    true_class: int
    predicted_class: int
    count: int


def top_confused_pairs(confusion: np.ndarray, top_n: int = 5) -> list[ConfusedPair]:
    """Off-diagonal confusion matrix entries, largest first."""
    pairs = []
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            if i != j and confusion[i, j] > 0:
                pairs.append(ConfusedPair(true_class=i, predicted_class=j, count=int(confusion[i, j])))
    pairs.sort(key=lambda p: p.count, reverse=True)
    return pairs[:top_n]


def pairwise_class_separability(features: np.ndarray, labels: np.ndarray, class_a: int, class_b: int) -> float:
    """Normalized distance between two classes' mean feature vectors (in units of their
    pooled per-channel std) - a quick, model-independent check for whether two classes
    are inherently similar in feature space, as opposed to a model simply failing to
    learn a real distinction. Small values support "these classes are genuinely similar
    in this feature space", regardless of which model is used.
    """
    feats_a = features[labels == class_a]
    feats_b = features[labels == class_b]
    if len(feats_a) == 0 or len(feats_b) == 0:
        raise ValueError("One of the classes has no examples in this feature set")

    mean_a, mean_b = feats_a.mean(axis=0), feats_b.mean(axis=0)
    pooled_std = np.concatenate([feats_a, feats_b], axis=0).std(axis=0) + 1e-6
    normalized_diff = np.abs(mean_a - mean_b) / pooled_std
    return float(normalized_diff.mean())
