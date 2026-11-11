import numpy as np
import pytest

from src.data.activity_dataset import (
    ACTIVITY_CLASSES,
    NUM_ACTIVITY_CLASSES,
    NUM_CHANNELS,
    activity_label_fn,
    activity_subject_wise_split,
    generate_synthetic_activity_dataset,
)


def test_generated_dataset_has_expected_size_and_shape():
    samples = generate_synthetic_activity_dataset(num_subjects=3, episodes_per_subject_per_class=2, seed=0)
    assert len(samples) == 3 * NUM_ACTIVITY_CLASSES * 2

    for s in samples:
        assert s.sequence.ndim == 2
        assert s.sequence.shape[1] == NUM_CHANNELS == 30
        assert s.sequence.shape[0] > 0
        assert 0 <= s.activity_id < NUM_ACTIVITY_CLASSES
        assert np.isfinite(s.sequence).all()


def test_all_activity_classes_represented():
    samples = generate_synthetic_activity_dataset(num_subjects=2, episodes_per_subject_per_class=1, seed=1)
    seen = {s.activity_id for s in samples}
    assert seen == set(range(len(ACTIVITY_CLASSES)))


def test_label_fn_matches_activity_id():
    samples = generate_synthetic_activity_dataset(num_subjects=1, episodes_per_subject_per_class=1, seed=2)
    for s in samples:
        assert activity_label_fn(s) == s.activity_id


def test_different_classes_are_not_identical():
    samples = generate_synthetic_activity_dataset(num_subjects=1, episodes_per_subject_per_class=1, seed=3)
    standing = next(s for s in samples if ACTIVITY_CLASSES[s.activity_id] == "standing")
    squatting = next(s for s in samples if ACTIVITY_CLASSES[s.activity_id] == "squatting")
    # knee landmarks should differ substantially between standing and squatting
    assert not np.allclose(standing.sequence.mean(axis=0), squatting.sequence.mean(axis=0), atol=0.05)


def test_subject_wise_split_disjoint():
    samples = generate_synthetic_activity_dataset(num_subjects=6, episodes_per_subject_per_class=1, seed=4)
    train, val, test = activity_subject_wise_split(samples, {1, 2, 3}, {4}, {5, 6})
    assert all(s.subject_id in {1, 2, 3} for s in train)
    assert all(s.subject_id in {4} for s in val)
    assert all(s.subject_id in {5, 6} for s in test)
    assert len(train) + len(val) + len(test) == len(samples)


def test_subject_wise_split_rejects_overlap():
    samples = generate_synthetic_activity_dataset(num_subjects=3, episodes_per_subject_per_class=1, seed=5)
    with pytest.raises(ValueError):
        activity_subject_wise_split(samples, {1}, {1}, {2})
