"""Shared evaluation metrics for the movement-classification models (XGBoost, LSTM, GRU),
so all three are scored identically and comparably."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


@dataclass
class ClassificationReport:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_f1: np.ndarray
    confusion: np.ndarray

    def summary(self) -> str:
        lines = [
            f"accuracy={self.accuracy:.4f}",
            f"macro_precision={self.macro_precision:.4f}",
            f"macro_recall={self.macro_recall:.4f}",
            f"macro_f1={self.macro_f1:.4f}",
        ]
        return " ".join(lines)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> ClassificationReport:
    labels = list(range(num_classes))
    accuracy = float((y_true == y_pred).mean())
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    macro_f1 = float(per_class_f1.mean())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return ClassificationReport(
        accuracy=accuracy,
        macro_precision=float(precision),
        macro_recall=float(recall),
        macro_f1=macro_f1,
        per_class_f1=per_class_f1,
        confusion=cm,
    )
