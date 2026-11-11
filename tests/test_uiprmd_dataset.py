import numpy as np
import pytest

from src.data.uiprmd_dataset import UIPRMDLoadError, load_all_episodes, load_episode, subject_wise_split


def _write_episode_file(path, n_frames: int, n_channels: int = 66) -> None:
    data = np.random.randn(n_frames, n_channels).astype(np.float32)
    np.savetxt(path, data, delimiter=",")


@pytest.fixture
def sample_angles_dir(tmp_path):
    for movement in (1, 2):
        for subject in (1, 2):
            for episode in (1,):
                name = f"m{movement:02d}_s{subject:02d}_e{episode:02d}_angles.txt"
                _write_episode_file(tmp_path / name, n_frames=10 + movement)
    return tmp_path


def test_load_episode_parses_filename_and_shape(sample_angles_dir):
    episode = load_episode(sample_angles_dir / "m01_s01_e01_angles.txt")
    assert episode.movement_id == 1
    assert episode.subject_id == 1
    assert episode.episode_id == 1
    assert episode.sequence.shape == (11, 66)


def test_load_episode_wrong_channel_count_raises(tmp_path):
    path = tmp_path / "m01_s01_e01_angles.txt"
    _write_episode_file(path, n_frames=5, n_channels=10)
    with pytest.raises(UIPRMDLoadError):
        load_episode(path)


def test_load_episode_bad_filename_raises(tmp_path):
    path = tmp_path / "not_a_uiprmd_file.txt"
    _write_episode_file(path, n_frames=5)
    with pytest.raises(UIPRMDLoadError):
        load_episode(path)


def test_load_all_episodes_finds_every_file(sample_angles_dir):
    episodes = load_all_episodes(sample_angles_dir)
    assert len(episodes) == 4  # 2 movements x 2 subjects x 1 episode


def test_load_all_episodes_empty_dir_raises(tmp_path):
    with pytest.raises(UIPRMDLoadError):
        load_all_episodes(tmp_path)


def test_subject_wise_split_is_disjoint_and_covers_data(sample_angles_dir):
    episodes = load_all_episodes(sample_angles_dir)
    train, val, test = subject_wise_split(episodes, {1}, {2}, set())

    assert all(e.subject_id == 1 for e in train)
    assert all(e.subject_id == 2 for e in val)
    assert test == []
    assert len(train) + len(val) == len(episodes)


def test_subject_wise_split_rejects_overlapping_subjects(sample_angles_dir):
    episodes = load_all_episodes(sample_angles_dir)
    with pytest.raises(ValueError):
        subject_wise_split(episodes, {1}, {1}, set())
