import numpy as np
import torch

from src.data.uiprmd_dataset import Episode
from src.models.sequence_models import GRUClassifier, LSTMClassifier
from src.training.train_sequence_model import predict_sequence_model, train_sequence_model

NUM_CLASSES = 3
INPUT_SIZE = 66


def _random_episodes(n: int, movement_ids: list[int], length: int = 20) -> list[Episode]:
    episodes = []
    for i in range(n):
        seq = np.random.randn(length, INPUT_SIZE).astype(np.float32)
        episodes.append(Episode(movement_id=movement_ids[i % len(movement_ids)], subject_id=(i % 4) + 1, episode_id=1, sequence=seq))
    return episodes


def test_lstm_forward_output_shape():
    model = LSTMClassifier(input_size=INPUT_SIZE, hidden_size=16, num_layers=1, num_classes=NUM_CLASSES, dropout=0.0)
    x = torch.randn(5, 20, INPUT_SIZE)
    out = model(x)
    assert out.shape == (5, NUM_CLASSES)


def test_gru_forward_output_shape():
    model = GRUClassifier(input_size=INPUT_SIZE, hidden_size=16, num_layers=2, num_classes=NUM_CLASSES, dropout=0.1)
    x = torch.randn(4, 15, INPUT_SIZE)
    out = model(x)
    assert out.shape == (4, NUM_CLASSES)


def test_train_and_predict_sequence_model_smoke():
    train_eps = _random_episodes(12, movement_ids=[1, 2, 3])
    val_eps = _random_episodes(6, movement_ids=[1, 2, 3])
    test_eps = _random_episodes(6, movement_ids=[1, 2, 3])

    model = GRUClassifier(input_size=INPUT_SIZE, hidden_size=8, num_layers=1, num_classes=NUM_CLASSES, dropout=0.0)
    fit_result = train_sequence_model(
        model, train_eps, val_eps, target_length=20, batch_size=4, epochs=2, learning_rate=1e-3, device="cpu"
    )
    assert "normalize_stats" in fit_result
    assert len(fit_result["history"]["train_loss"]) >= 1

    y_true, y_pred = predict_sequence_model(model, test_eps, target_length=20, normalize_stats=fit_result["normalize_stats"], device="cpu")
    assert y_true.shape == y_pred.shape == (6,)
    assert set(np.unique(y_pred)).issubset({0, 1, 2})
