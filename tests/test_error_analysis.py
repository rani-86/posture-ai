import numpy as np

from src.evaluation.error_analysis import pairwise_class_separability, top_confused_pairs


def test_top_confused_pairs_ignores_diagonal():
    confusion = np.array([[10, 2, 0], [1, 8, 3], [0, 0, 9]])
    pairs = top_confused_pairs(confusion, top_n=5)
    assert all(p.true_class != p.predicted_class for p in pairs)
    assert pairs[0].count == 3  # the largest off-diagonal entry
    assert pairs[0].true_class == 1 and pairs[0].predicted_class == 2


def test_top_confused_pairs_respects_top_n():
    confusion = np.array([[5, 4], [3, 6]])
    pairs = top_confused_pairs(confusion, top_n=1)
    assert len(pairs) == 1


def test_pairwise_class_separability_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    features = rng.normal(0, 1, size=(100, 5))
    labels = np.array([0] * 50 + [1] * 50)
    # Same distribution for both classes -> means should be close -> low separability
    score = pairwise_class_separability(features, labels, 0, 1)
    assert score < 0.5


def test_pairwise_class_separability_larger_than_identical_distributions():
    identical_features = np.random.default_rng(1).normal(0, 1, size=(100, 3))
    identical_labels = np.array([0] * 50 + [1] * 50)
    low_score = pairwise_class_separability(identical_features, identical_labels, 0, 1)

    separated_features = np.concatenate([np.zeros((50, 3)), np.full((50, 3), 10.0)])
    separated_labels = np.array([0] * 50 + [1] * 50)
    high_score = pairwise_class_separability(separated_features, separated_labels, 0, 1)

    assert high_score > low_score
