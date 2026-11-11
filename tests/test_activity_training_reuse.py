"""Confirms the Phase 4 training/dataset code generalizes cleanly to Phase 5's activity
samples via label_fn, without needing separate duplicated training loops."""
from src.data.activity_dataset import activity_label_fn, generate_synthetic_activity_dataset
from src.models.sequence_models import GRUClassifier
from src.training.train_sequence_model import predict_sequence_model, train_sequence_model
from src.training.train_xgboost import build_feature_matrix, train_xgboost


def test_sequence_model_trains_on_activity_samples():
    samples = generate_synthetic_activity_dataset(num_subjects=4, episodes_per_subject_per_class=2, seed=10)
    train_s = [s for s in samples if s.subject_id in (1, 2)]
    val_s = [s for s in samples if s.subject_id == 3]
    test_s = [s for s in samples if s.subject_id == 4]

    model = GRUClassifier(input_size=30, hidden_size=8, num_layers=1, num_classes=5, dropout=0.0)
    fit = train_sequence_model(
        model, train_s, val_s, target_length=60, batch_size=8, epochs=2, learning_rate=1e-3,
        device="cpu", label_fn=activity_label_fn,
    )
    y_true, y_pred = predict_sequence_model(
        model, test_s, target_length=60, normalize_stats=fit["normalize_stats"], device="cpu",
        label_fn=activity_label_fn,
    )
    assert y_true.shape == y_pred.shape == (len(test_s),)
    assert set(y_true).issubset(set(range(5)))


def test_xgboost_trains_on_activity_samples():
    samples = generate_synthetic_activity_dataset(num_subjects=4, episodes_per_subject_per_class=2, seed=11)
    train_s = [s for s in samples if s.subject_id in (1, 2, 3)]
    val_s = [s for s in samples if s.subject_id == 4]

    model = train_xgboost(train_s, val_s, num_classes=5, seed=42, label_fn=activity_label_fn)
    x_val, y_val = build_feature_matrix(val_s, label_fn=activity_label_fn)
    preds = model.predict(x_val)
    assert preds.shape == y_val.shape
