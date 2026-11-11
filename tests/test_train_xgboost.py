import numpy as np

from src.data.uiprmd_dataset import Episode
from src.training.train_xgboost import build_feature_matrix, train_xgboost


def _episodes(n: int, movement_ids: list[int]) -> list[Episode]:
    out = []
    for i in range(n):
        seq = np.random.randn(15, 66).astype(np.float32)
        out.append(Episode(movement_id=movement_ids[i % len(movement_ids)], subject_id=(i % 3) + 1, episode_id=1, sequence=seq))
    return out


def test_build_feature_matrix_shape():
    episodes = _episodes(5, [1, 2])
    x, y = build_feature_matrix(episodes)
    assert x.shape == (5, 66 * 5)  # mean/std/min/max/range per channel
    assert y.shape == (5,)
    assert set(y).issubset({0, 1})  # 0-indexed


def test_train_xgboost_fits_and_predicts():
    train_eps = _episodes(20, [1, 2, 3])
    val_eps = _episodes(9, [1, 2, 3])
    model = train_xgboost(train_eps, val_eps, num_classes=3, seed=42)

    x_val, y_val = build_feature_matrix(val_eps)
    preds = model.predict(x_val)
    assert preds.shape == y_val.shape
    assert set(preds).issubset({0, 1, 2})
