"""Trains an XGBoost multi-class classifier on per-sample summary-statistic features.

Shared by Phase 4 (UI-PRMD movement classification) and Phase 5 (activity recognition)
via the `label_fn` passed into `build_feature_matrix`.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import xgboost as xgb

from src.data.sequence_dataset import uiprmd_label_fn
from src.feature_extraction.sequence_features import summary_statistics_features


def build_feature_matrix(samples: list, label_fn: Callable = uiprmd_label_fn) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([summary_statistics_features(s) for s in samples])
    y = np.array([label_fn(s) for s in samples])
    return x, y


def train_xgboost(
    train_episodes: list,
    val_episodes: list,
    num_classes: int,
    seed: int,
    label_fn: Callable = uiprmd_label_fn,
) -> xgb.XGBClassifier:
    x_train, y_train = build_feature_matrix(train_episodes, label_fn)
    x_val, y_val = build_feature_matrix(val_episodes, label_fn)

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
