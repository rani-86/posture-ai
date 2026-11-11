"""Phase 6 evaluation entrypoint: loads already-trained checkpoints (does NOT retrain)
and runs consolidated evaluation + error analysis across the Phase 4 (UI-PRMD movement
classification) and Phase 5 (synthetic activity recognition) models, plus a Phase 2
video-pipeline throughput benchmark.

Keeping this separate from train.py/train_activity.py reflects a real ML-engineering
boundary: training is expensive and should be deliberate; evaluation/inference should be
cheap, repeatable, and runnable against a fixed checkpoint without retraining it.

Usage:
    .venv/Scripts/python.exe evaluate.py
"""
from __future__ import annotations

import json

import numpy as np
import torch
import xgboost as xgb

from src.data.activity_dataset import (
    NUM_ACTIVITY_CLASSES,
    activity_label_fn,
    activity_subject_wise_split,
    generate_synthetic_activity_dataset,
)
from src.data.uiprmd_dataset import NUM_MOVEMENT_CLASSES, load_all_episodes, subject_wise_split
from src.evaluation.classification_metrics import evaluate_predictions
from src.evaluation.error_analysis import top_confused_pairs
from src.evaluation.pipeline_benchmark import benchmark_pipeline
from src.models.sequence_models import GRUClassifier, LSTMClassifier
from src.preprocessing.video_io import AnnotatedVideoWriter
from src.training.train_sequence_model import predict_sequence_model
from src.training.train_xgboost import build_feature_matrix
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from train import TEST_SUBJECTS as UIPRMD_TEST_SUBJECTS
from train import TRAIN_SUBJECTS as UIPRMD_TRAIN_SUBJECTS
from train import UIPRMD_ANGLES_DIR, VAL_SUBJECTS as UIPRMD_VAL_SUBJECTS
from train import TARGET_SEQUENCE_LENGTH as UIPRMD_TARGET_LENGTH
from train_activity import NUM_SUBJECTS as ACTIVITY_NUM_SUBJECTS
from train_activity import TARGET_SEQUENCE_LENGTH as ACTIVITY_TARGET_LENGTH
from train_activity import TEST_SUBJECTS as ACTIVITY_TEST_SUBJECTS
from train_activity import TRAIN_SUBJECTS as ACTIVITY_TRAIN_SUBJECTS
from train_activity import VAL_SUBJECTS as ACTIVITY_VAL_SUBJECTS

logger = get_logger(__name__)


def _load_normalize_stats(path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["arr_0"], data["arr_1"]


def evaluate_uiprmd_models(config) -> dict:
    episodes = load_all_episodes(UIPRMD_ANGLES_DIR)
    _, _, test_eps = subject_wise_split(episodes, UIPRMD_TRAIN_SUBJECTS, UIPRMD_VAL_SUBJECTS, UIPRMD_TEST_SUBJECTS)

    results = {}

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(config.checkpoints_dir / "xgboost_movement_classifier.json"))
    x_test, y_test = build_feature_matrix(test_eps)
    results["xgboost"] = evaluate_predictions(y_test, xgb_model.predict(x_test), NUM_MOVEMENT_CLASSES)

    for name, cls in (("lstm", LSTMClassifier), ("gru", GRUClassifier)):
        model = cls(
            input_size=66,
            hidden_size=config.sequence_model.hidden_size,
            num_layers=config.sequence_model.num_layers,
            num_classes=NUM_MOVEMENT_CLASSES,
            dropout=config.sequence_model.dropout,
        )
        model.load_state_dict(torch.load(config.checkpoints_dir / f"{name}_movement_classifier.pt"))
        stats = _load_normalize_stats(config.checkpoints_dir / f"{name}_movement_classifier_normalize_stats.npz")
        y_true, y_pred = predict_sequence_model(model, test_eps, UIPRMD_TARGET_LENGTH, stats, "cpu")
        results[name] = evaluate_predictions(y_true, y_pred, NUM_MOVEMENT_CLASSES)

    return results


def evaluate_activity_models(config) -> dict:
    samples = generate_synthetic_activity_dataset(num_subjects=ACTIVITY_NUM_SUBJECTS, seed=config.seed)
    _, _, test_s = activity_subject_wise_split(samples, ACTIVITY_TRAIN_SUBJECTS, ACTIVITY_VAL_SUBJECTS, ACTIVITY_TEST_SUBJECTS)

    results = {}

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(config.checkpoints_dir / "xgboost_activity_classifier.json"))
    x_test, y_test = build_feature_matrix(test_s, label_fn=activity_label_fn)
    results["xgboost"] = evaluate_predictions(y_test, xgb_model.predict(x_test), NUM_ACTIVITY_CLASSES)

    for name, cls in (("lstm", LSTMClassifier), ("gru", GRUClassifier)):
        model = cls(
            input_size=30,
            hidden_size=config.sequence_model.hidden_size,
            num_layers=config.sequence_model.num_layers,
            num_classes=NUM_ACTIVITY_CLASSES,
            dropout=config.sequence_model.dropout,
        )
        model.load_state_dict(torch.load(config.checkpoints_dir / f"{name}_activity_classifier.pt"))
        stats = _load_normalize_stats(config.checkpoints_dir / f"{name}_activity_classifier_normalize_stats.npz")
        y_true, y_pred = predict_sequence_model(
            model, test_s, ACTIVITY_TARGET_LENGTH, stats, "cpu", label_fn=activity_label_fn
        )
        results[name] = evaluate_predictions(y_true, y_pred, NUM_ACTIVITY_CLASSES)

    return results


def main() -> None:
    config = load_config()

    logger.info("=== Re-evaluating Phase 4 (UI-PRMD movement classification) checkpoints ===")
    uiprmd_results = evaluate_uiprmd_models(config)
    for name, report in uiprmd_results.items():
        logger.info(f"{name:10s} {report.summary()}")
        confused = top_confused_pairs(report.confusion, top_n=3)
        for pair in confused:
            logger.info(f"  confused: true=m{pair.true_class + 1:02d} predicted=m{pair.predicted_class + 1:02d} count={pair.count}")

    logger.info("=== Re-evaluating Phase 5 (synthetic activity recognition) checkpoints ===")
    activity_results = evaluate_activity_models(config)
    for name, report in activity_results.items():
        logger.info(f"{name:10s} {report.summary()}")

    logger.info("=== Phase 2 video-pipeline throughput benchmark (synthetic video) ===")
    import cv2
    import numpy as np

    config.raw_data_dir.mkdir(parents=True, exist_ok=True)
    bench_video_path = config.raw_data_dir / "benchmark_synthetic.mp4"
    fps, w, h, n = 15, 320, 240, 150
    writer = cv2.VideoWriter(str(bench_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        writer.write(np.full((h, w, 3), fill_value=(i * 5) % 256, dtype=np.uint8))
    writer.release()

    benchmark = benchmark_pipeline(bench_video_path, config)
    logger.info(
        f"Processed {benchmark.num_frames} frames in {benchmark.wall_clock_sec:.2f}s "
        f"({benchmark.fps:.1f} fps), peak Python-heap memory {benchmark.peak_memory_mb:.1f} MB"
    )
    bench_video_path.unlink()

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "uiprmd_movement_classification": {
            name: {"accuracy": r.accuracy, "macro_f1": r.macro_f1} for name, r in uiprmd_results.items()
        },
        "synthetic_activity_recognition": {
            name: {"accuracy": r.accuracy, "macro_f1": r.macro_f1} for name, r in activity_results.items()
        },
        "video_pipeline_benchmark": {
            "num_frames": benchmark.num_frames,
            "wall_clock_sec": benchmark.wall_clock_sec,
            "fps": benchmark.fps,
            "peak_memory_mb": benchmark.peak_memory_mb,
        },
    }
    report_path = config.reports_dir / "phase6_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Saved consolidated evaluation report to {report_path}")


if __name__ == "__main__":
    main()
