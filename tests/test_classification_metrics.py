import numpy as np

from src.evaluation.classification_metrics import evaluate_predictions


def test_perfect_predictions_give_perfect_scores():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = y_true.copy()
    report = evaluate_predictions(y_true, y_pred, num_classes=3)

    assert report.accuracy == 1.0
    assert report.macro_precision == 1.0
    assert report.macro_recall == 1.0
    assert report.macro_f1 == 1.0
    assert np.array_equal(report.confusion, np.eye(3, dtype=int) * 2)


def test_all_wrong_predictions_give_zero_accuracy():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    report = evaluate_predictions(y_true, y_pred, num_classes=2)
    assert report.accuracy == 0.0


def test_confusion_matrix_shape_matches_num_classes():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 2, 1])
    report = evaluate_predictions(y_true, y_pred, num_classes=3)
    assert report.confusion.shape == (3, 3)
    assert report.per_class_f1.shape == (3,)
