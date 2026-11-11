"""Trains an XGBoost multi-class classifier on per-episode summary-statistic features."""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from src.data.uiprmd_dataset import Episode
from src.feature_extraction.sequence_features import summary_statistics_features


def build_feature_matrix(episodes: list[Episode]) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([summary_statistics_features(e) for e in episodes])
    y = np.array([e.movement_id - 1 for e in episodes])  # 0-indexed
    return x, y


def train_xgboost(
    train_episodes: list[Episode],
    val_episodes: list[Episode],
    num_classes: int,
    seed: int,
) -> xgb.XGBClassifier:
    x_train, y_train = build_feature_matrix(train_episodes)
    x_val, y_val = build_feature_matrix(val_episodes)

    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=num_classes,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=seed,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model
