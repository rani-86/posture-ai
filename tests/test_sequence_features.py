import numpy as np

from src.data.uiprmd_dataset import Episode
from src.feature_extraction.sequence_features import pad_or_truncate_sequence, summary_statistics_features


def _episode(n_frames: int, n_channels: int = 66) -> Episode:
    seq = np.random.randn(n_frames, n_channels).astype(np.float32)
    return Episode(movement_id=1, subject_id=1, episode_id=1, sequence=seq)


def test_summary_statistics_features_shape_and_values():
    seq = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]], dtype=np.float32)
    episode = Episode(movement_id=1, subject_id=1, episode_id=1, sequence=seq)

    features = summary_statistics_features(episode)
    assert features.shape == (10,)  # 2 channels x 5 stats (mean, std, min, max, range)

    mean, std, minimum, maximum, value_range = features.reshape(5, 2)
    assert np.allclose(mean, [3.0, 20.0])
    assert np.allclose(minimum, [1.0, 10.0])
    assert np.allclose(maximum, [5.0, 30.0])
    assert np.allclose(value_range, [4.0, 20.0])


def test_pad_or_truncate_pads_with_last_frame():
    seq = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    out = pad_or_truncate_sequence(seq, target_length=4)
    assert out.shape == (4, 2)
    assert np.allclose(out[2:], [[2.0, 2.0], [2.0, 2.0]])  # last frame repeated


def test_pad_or_truncate_truncates():
    seq = np.arange(10).reshape(5, 2).astype(np.float32)
    out = pad_or_truncate_sequence(seq, target_length=3)
    assert out.shape == (3, 2)
    assert np.array_equal(out, seq[:3])


def test_pad_or_truncate_exact_length_unchanged():
    seq = _episode(7).sequence
    out = pad_or_truncate_sequence(seq, target_length=7)
    assert np.array_equal(out, seq)
