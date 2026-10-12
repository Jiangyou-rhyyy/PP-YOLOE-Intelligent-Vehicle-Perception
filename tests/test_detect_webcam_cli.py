from __future__ import annotations

import numpy as np
import pytest


class FakeDetector:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def predict(self, image: np.ndarray) -> list:
        return []


class FakeCapture:
    def __init__(self, frames, opened: bool = True) -> None:
        self.frames = list(frames)
        self.opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self):
        if not self.frames:
            self.opened = False
            return False, None
        return self.frames.pop(0)

    def release(self) -> None:
        self.released = True


class FailingAfterInitialFrameCapture(FakeCapture):
    def __init__(self, frame: np.ndarray) -> None:
        super().__init__([(True, frame)])
        self.failed_reads = 0

    def read(self):
        if self.frames:
            return self.frames.pop(0)
        self.failed_reads += 1
        if self.failed_reads > 3:
            self.opened = False
        return False, None


def configure_runtime(monkeypatch, module, capture: FakeCapture, keys: list[int]) -> dict:
    state = {"destroyed": False, "shown": 0}
    monkeypatch.setattr(module, "OpenVINODetector", FakeDetector)
    monkeypatch.setattr(module, "load_labels", lambda path: ["car"])
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda camera: capture)
    monkeypatch.setattr(module.cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(module.cv2, "resizeWindow", lambda *args: None)
    monkeypatch.setattr(module.cv2, "imshow", lambda *args: state.__setitem__("shown", state["shown"] + 1))
    monkeypatch.setattr(module.cv2, "waitKey", lambda delay: keys.pop(0) if keys else 27)
    monkeypatch.setattr(module.cv2, "destroyAllWindows", lambda: state.__setitem__("destroyed", True))
    return state


def test_webcam_cli_help() -> None:
    from scripts import detect_webcam

    with pytest.raises(SystemExit) as raised:
        detect_webcam.main(["--help"])
    assert raised.value.code == 0


def test_camera_open_failure_releases_resources(tmp_path, monkeypatch, capsys) -> None:
    from scripts import detect_webcam

    capture = FakeCapture([], opened=False)
    state = configure_runtime(monkeypatch, detect_webcam, capture, [])

    result = detect_webcam.main(
        ["--model", str(tmp_path / "model.xml"), "--labels", str(tmp_path / "labels.txt")]
    )

    assert result != 0
    assert capture.released
    assert state["destroyed"]
    assert "Unable to open camera" in capsys.readouterr().err


def test_dropped_frame_is_skipped_and_escape_stops_cleanly(tmp_path, monkeypatch) -> None:
    from scripts import detect_webcam

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    capture = FakeCapture([(True, frame), (False, None), (True, frame)])
    state = configure_runtime(monkeypatch, detect_webcam, capture, [27])

    result = detect_webcam.main(
        [
            "--model",
            str(tmp_path / "model.xml"),
            "--labels",
            str(tmp_path / "labels.txt"),
            "--window-size",
            "640x480",
        ]
    )

    assert result == 0
    assert state["shown"] == 1
    assert capture.released
    assert state["destroyed"]


def test_persistent_dropped_frames_return_an_error_and_release_resources(
    tmp_path, monkeypatch, capsys
) -> None:
    from scripts import detect_webcam

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    capture = FailingAfterInitialFrameCapture(frame)
    state = configure_runtime(monkeypatch, detect_webcam, capture, [0, 0, 0, 0])
    monkeypatch.setattr(
        detect_webcam, "MAX_CONSECUTIVE_DROPPED_FRAMES", 2, raising=False
    )

    result = detect_webcam.main(
        ["--model", str(tmp_path / "model.xml"), "--labels", str(tmp_path / "labels.txt")]
    )

    assert result != 0
    assert capture.released
    assert state["destroyed"]
    assert "Camera stopped returning frames" in capsys.readouterr().err
