import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_video(tmp_path):
    """A short synthetic video: solid-color frames at a known fps/size/frame count.

    Not a real human video - used to validate video I/O mechanics (frame count, fps,
    timestamps, resampling) independently of pose detection, which needs a real person.
    """
    path = tmp_path / "synthetic.mp4"
    fps, width, height, n_frames = 30, 64, 48, 60

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), fill_value=i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    return {"path": path, "fps": fps, "width": width, "height": height, "n_frames": n_frames}
