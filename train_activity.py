"""Phase 5 training entrypoint: trains XGBoost, LSTM and GRU activity-recognition
classifiers (sitting / standing / walking / bending / squatting).

IMPORTANT: trained and evaluated on SYNTHETIC, procedurally-generated pose sequences
(src/data/activity_dataset.py) - no real labeled video clips were available. This proves
the pipeline works end-to-end in the same feature space the real MediaPipe pipeline
(Phase 2/3) produces; it does NOT measure real-world activity-recognition accuracy. See
docs/phase5_notes.md before interpreting any number this script prints.

Usage:
    .venv/Scripts/python.exe train_activity.py
"""
from __future__ import annotations

import json

import numpy as np
import torch

from src.data.activity_dataset import (
    NUM_ACTIVITY_CLASSES,
    activity_label_fn,
    activity_subject_wise_split,
    generate_synthetic_activity_dataset,
)
from src.evaluation.classification_metrics import evaluate_predictions
from src.models.sequence_models import GRUClassifier, LSTMClassifier
from src.training.train_sequence_model import predict_sequence_model, train_sequence_model
from src.training.train_xgboost import build_feature_matrix, train_xgboost
from src.utils.config import load_config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

TARGET_SEQUENCE_LENGTH = 90
NUM_SUBJECTS = 20
TRAIN_SUBJECTS = set(range(1, 15))  # 1-14
VAL_SUBJECTS = set(range(15, 18))  # 15-17
TEST_SUBJECTS = set(range(18, 21))  # 18-20


def main() -> None:
    config = load_config()
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() and config.device != "cpu" else "cpu"
    logger.info(f"Using device: {device}")
    logger.warning("Training on SYNTHETIC placeholder activity data - see docs/phase5_notes.md")

    samples = generate_synthetic_activity_dataset(num_subjects=NUM_SUBJECTS, seed=config.seed)
    train_s, val_s, test_s = activity_subject_wise_split(samples, TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS)
    logger.info(f"Split sizes: train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    results = {}

    logger.info("Training XGBoost...")
    xgb_model = train_xgboost(train_s, val_s, NUM_ACTIVITY_CLASSES, config.seed, label_fn=activity_label_fn)
    x_test, y_test = build_feature_matrix(test_s, label_fn=activity_label_fn)
    xgb_preds = xgb_model.predict(x_test)
    xgb_report = evaluate_predictions(y_test, xgb_preds, NUM_ACTIVITY_CLASSES)
    logger.info(f"XGBoost test: {xgb_report.summary()}")
    results["xgboost"] = xgb_report
    config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    xgb_model.save_model(str(config.checkpoints_dir / "xgboost_activity_classifier.json"))

    for name, model_cls in (("lstm", LSTMClassifier), ("gru", GRUClassifier)):
        logger.info(f"Training {name.upper()}...")
        model = model_cls(
            input_size=30,
            hidden_size=config.sequence_model.hidden_size,
            num_layers=config.sequence_model.num_layers,
            num_classes=NUM_ACTIVITY_CLASSES,
            dropout=config.sequence_model.dropout,
        )
        fit = train_sequence_model(
            model, train_s, val_s, TARGET_SEQUENCE_LENGTH,
            config.training.batch_size, config.training.epochs, config.training.learning_rate, device,
            label_fn=activity_label_fn,
        )
        y_true, y_pred = predict_sequence_model(
            model, test_s, TARGET_SEQUENCE_LENGTH, fit["normalize_stats"], device, label_fn=activity_label_fn
        )
        report = evaluate_predictions(y_true, y_pred, NUM_ACTIVITY_CLASSES)
        logger.info(f"{name.upper()} test: {report.summary()}")
        results[name] = report
        torch.save(model.state_dict(), config.checkpoints_dir / f"{name}_activity_classifier.pt")
        np.savez(config.checkpoints_dir / f"{name}_activity_classifier_normalize_stats.npz", *fit["normalize_stats"])

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.reports_dir / "phase5_activity_model_comparison.json"
    serializable = {
        name: {
            "accuracy": r.accuracy,
            "macro_precision": r.macro_precision,
            "macro_recall": r.macro_recall,
            "macro_f1": r.macro_f1,
            "per_class_f1": r.per_class_f1.tolist(),
            "confusion_matrix": r.confusion.tolist(),
        }
        for name, r in results.items()
    }
    report_path.write_text(json.dumps(serializable, indent=2))
    logger.info(f"Saved comparison report to {report_path}")

    print("\n=== Phase 5 activity model comparison (test set, SYNTHETIC data) ===")
    for name, r in results.items():
        print(f"{name:10s} {r.summary()}")


if __name__ == "__main__":
    main()
