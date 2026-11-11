"""Loads trained model checkpoints once and keeps them in memory for reuse across
requests - inference code should never re-read a checkpoint file per request.

Only the Phase 5 activity classifier is loaded for live inference. The Phase 4 UI-PRMD
movement classifier is deliberately NOT wired into the API: it was trained on UI-PRMD's
66-channel Kinect Euler-angle feature space, which is not the same feature space this
project's own MediaPipe pipeline produces (documented in docs/phase4_notes.md) - calling
it on real pipeline output would silently feed it meaningless numbers in the right shape,
which is worse than not exposing it at all.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from src.models.sequence_models import LSTMClassifier
from src.utils.config import load_config


class ActivityModelBundle:
    def __init__(self, model: LSTMClassifier, normalize_stats: tuple[np.ndarray, np.ndarray], target_length: int):
        self.model = model
        self.mean, self.std = normalize_stats
        self.target_length = target_length
        self.model.eval()

    def predict(self, sequence: np.ndarray) -> tuple[int, np.ndarray]:
        """`sequence` is (T, 30) raw landmark channels. Returns (predicted_class, probabilities)."""
        from src.feature_extraction.sequence_features import pad_or_truncate_sequence

        padded = pad_or_truncate_sequence(sequence, self.target_length)
        normalized = (padded - self.mean) / self.std
        x = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)  # (1, T, 30)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
        return int(probs.argmax()), probs


@lru_cache(maxsize=1)
def get_activity_model() -> ActivityModelBundle:
    # No config argument: PostureAIConfig holds an unhashable dict (`raw`), which would
    # break lru_cache's key hashing. This is a process-wide singleton loaded from the
    # default config path, which matches how the app actually uses it (one process, one
    # config file) - not meant to be swappable per-call.
    config = load_config()
    model_path = config.checkpoints_dir / "lstm_activity_classifier.pt"
    stats_path = config.checkpoints_dir / "lstm_activity_classifier_normalize_stats.npz"

    if not model_path.exists() or not stats_path.exists():
        raise FileNotFoundError(
            f"Activity model checkpoint not found at {model_path}. "
            "Run train_activity.py first (see docs/phase5_notes.md)."
        )

    model = LSTMClassifier(
        input_size=30,
        hidden_size=config.sequence_model.hidden_size,
        num_layers=config.sequence_model.num_layers,
        num_classes=5,
        dropout=config.sequence_model.dropout,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    stats_data = np.load(stats_path)
    normalize_stats = (stats_data["arr_0"], stats_data["arr_1"])

    from train_activity import TARGET_SEQUENCE_LENGTH

    return ActivityModelBundle(model, normalize_stats, TARGET_SEQUENCE_LENGTH)
