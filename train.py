"""Phase 4 training entrypoint: trains XGBoost, LSTM and GRU movement-type classifiers
on UI-PRMD (Kinect, segmented-by-repetition angle sequences) and compares them.

Usage:
    .venv/Scripts/python.exe train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from src.data.uiprmd_dataset import NUM_MOVEMENT_CLASSES, load_all_episodes, subject_wise_split
from src.evaluation.classification_metrics import evaluate_predictions
from src.models.sequence_models import GRUClassifier, LSTMClassifier
from src.training.train_sequence_model import predict_sequence_model, train_sequence_model
from src.training.train_xgboost import build_feature_matrix, train_xgboost
from src.utils.config import load_config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

UIPRMD_ANGLES_DIR = "data/raw/UI-PRMD/Segmented Movements/Kinect/Angles"
TARGET_SEQUENCE_LENGTH = 100  # covers ~90th percentile of episode lengths without excessive padding

# Subject-wise split: no subject's repetitions appear in more than one split.
TRAIN_SUBJECTS = {1, 2, 3, 4, 5, 6, 7}
VAL_SUBJECTS = {8}
TEST_SUBJECTS = {9, 10}


def main() -> None:
    config = load_config()
    torch.manual_seed(config.seed)

    device = "cuda" if torch.cuda.is_available() and config.device != "cpu" else "cpu"
    logger.info(f"Using device: {device}")

    episodes = load_all_episodes(UIPRMD_ANGLES_DIR)
    train_eps, val_eps, test_eps = subject_wise_split(episodes, TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS)
    logger.info(f"Split sizes: train={len(train_eps)} val={len(val_eps)} test={len(test_eps)}")

    results = {}

    # --- XGBoost on summary-statistic features ---
    logger.info("Training XGBoost...")
    xgb_model = train_xgboost(train_eps, val_eps, NUM_MOVEMENT_CLASSES, config.seed)
    x_test, y_test = build_feature_matrix(test_eps)
    xgb_preds = xgb_model.predict(x_test)
    xgb_report = evaluate_predictions(y_test, xgb_preds, NUM_MOVEMENT_CLASSES)
    logger.info(f"XGBoost test: {xgb_report.summary()}")
    results["xgboost"] = xgb_report

    config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    xgb_model.save_model(str(config.checkpoints_dir / "xgboost_movement_classifier.json"))

    # --- LSTM ---
    logger.info("Training LSTM...")
    lstm = LSTMClassifier(
        input_size=66,
        hidden_size=config.sequence_model.hidden_size,
        num_layers=config.sequence_model.num_layers,
        num_classes=NUM_MOVEMENT_CLASSES,
        dropout=config.sequence_model.dropout,
    )
    lstm_fit = train_sequence_model(
        lstm, train_eps, val_eps, TARGET_SEQUENCE_LENGTH,
        config.training.batch_size, config.training.epochs, config.training.learning_rate, device,
    )
    y_true, y_pred = predict_sequence_model(lstm, test_eps, TARGET_SEQUENCE_LENGTH, lstm_fit["normalize_stats"], device)
    lstm_report = evaluate_predictions(y_true, y_pred, NUM_MOVEMENT_CLASSES)
    logger.info(f"LSTM test: {lstm_report.summary()}")
    results["lstm"] = lstm_report
    torch.save(lstm.state_dict(), config.checkpoints_dir / "lstm_movement_classifier.pt")

    # --- GRU ---
    logger.info("Training GRU...")
    gru = GRUClassifier(
        input_size=66,
        hidden_size=config.sequence_model.hidden_size,
        num_layers=config.sequence_model.num_layers,
        num_classes=NUM_MOVEMENT_CLASSES,
        dropout=config.sequence_model.dropout,
    )
    gru_fit = train_sequence_model(
        gru, train_eps, val_eps, TARGET_SEQUENCE_LENGTH,
        config.training.batch_size, config.training.epochs, config.training.learning_rate, device,
    )
    y_true, y_pred = predict_sequence_model(gru, test_eps, TARGET_SEQUENCE_LENGTH, gru_fit["normalize_stats"], device)
    gru_report = evaluate_predictions(y_true, y_pred, NUM_MOVEMENT_CLASSES)
    logger.info(f"GRU test: {gru_report.summary()}")
    results["gru"] = gru_report
    torch.save(gru.state_dict(), config.checkpoints_dir / "gru_movement_classifier.pt")

    # --- Save comparison report ---
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.reports_dir / "phase4_model_comparison.json"
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

    print("\n=== Phase 4 model comparison (test set, real UI-PRMD data) ===")
    for name, r in results.items():
        print(f"{name:10s} {r.summary()}")


if __name__ == "__main__":
    main()
