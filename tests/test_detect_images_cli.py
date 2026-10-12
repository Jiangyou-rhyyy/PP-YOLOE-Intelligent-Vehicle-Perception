from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from vehicle_perception.types import Detection


class FakeDetector:
    def __init__(self, *args, **kwargs) -> None:
        self.calls = 0

    def predict(self, image: np.ndarray) -> list[Detection]:
        self.calls += 1
        return [Detection(6, "car", 0.9, (1, 2, 8, 9))]


def configure_runtime(monkeypatch, module) -> None:
    monkeypatch.setattr(module, "OpenVINODetector", FakeDetector)
    monkeypatch.setattr(module, "load_labels", lambda path: ["car"])
    monkeypatch.setattr(
        module.cv2, "imread", lambda path: np.zeros((10, 10, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(module.cv2, "imwrite", lambda path, image: Path(path).write_bytes(b"jpg") or True)


def test_image_cli_help() -> None:
    from scripts import detect_images

    with pytest.raises(SystemExit) as raised:
        detect_images.main(["--help"])
    assert raised.value.code == 0


def test_image_cli_requires_image_or_directory(tmp_path) -> None:
    from scripts import detect_images

    with pytest.raises(SystemExit) as raised:
        detect_images.main(
            [
                "--model",
                str(tmp_path / "model.xml"),
                "--labels",
                str(tmp_path / "labels.txt"),
            ]
        )
    assert raised.value.code == 2


def test_single_image_creates_output_and_json_directories(tmp_path, monkeypatch) -> None:
    from scripts import detect_images

    configure_runtime(monkeypatch, detect_images)
    image_path = tmp_path / "scene.jpg"
    image_path.write_bytes(b"source")
    output_dir = tmp_path / "nested" / "images"
    json_dir = tmp_path / "nested" / "json"

    result = detect_images.main(
        [
            "--model",
            str(tmp_path / "model.xml"),
            "--labels",
            str(tmp_path / "labels.txt"),
            "--image",
            str(image_path),
            "--output-dir",
            str(output_dir),
            "--json-dir",
            str(json_dir),
        ]
    )

    assert result == 0
    assert (output_dir / "scene.jpg").is_file()
    payload = json.loads((json_dir / "scene.json").read_text(encoding="utf-8"))
    assert payload["frames"][0]["objects"][0]["category"] == "car"


def test_directory_mode_processes_supported_images_only(tmp_path, monkeypatch) -> None:
    from scripts import detect_images

    configure_runtime(monkeypatch, detect_images)
    image_dir = tmp_path / "input"
    image_dir.mkdir()
    (image_dir / "a.jpg").write_bytes(b"a")
    (image_dir / "b.png").write_bytes(b"b")
    (image_dir / "notes.txt").write_text("skip", encoding="utf-8")
    output_dir = tmp_path / "output"

    assert (
        detect_images.main(
            [
                "--model",
                str(tmp_path / "model.xml"),
                "--labels",
                str(tmp_path / "labels.txt"),
                "--image-dir",
                str(image_dir),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    assert sorted(path.name for path in output_dir.iterdir()) == ["a.jpg", "b.png"]


def test_unreadable_image_returns_nonzero(tmp_path, monkeypatch, capsys) -> None:
    from scripts import detect_images

    monkeypatch.setattr(detect_images, "OpenVINODetector", FakeDetector)
    monkeypatch.setattr(detect_images, "load_labels", lambda path: ["car"])
    monkeypatch.setattr(detect_images.cv2, "imread", lambda path: None)
    image_path = tmp_path / "broken.jpg"
    image_path.write_bytes(b"broken")

    result = detect_images.main(
        [
            "--model",
            str(tmp_path / "model.xml"),
            "--labels",
            str(tmp_path / "labels.txt"),
            "--image",
            str(image_path),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert result != 0
    assert "Unable to read image" in capsys.readouterr().err


def test_absolute_paths_work_outside_repository(tmp_path, monkeypatch) -> None:
    from scripts import detect_images

    configure_runtime(monkeypatch, detect_images)
    image_path = tmp_path / "scene.jpg"
    image_path.write_bytes(b"source")
    output_dir = tmp_path / "output"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    previous = Path.cwd()
    try:
        os.chdir(elsewhere)
        result = detect_images.main(
            [
                "--model",
                str((tmp_path / "model.xml").resolve()),
                "--labels",
                str((tmp_path / "labels.txt").resolve()),
                "--image",
                str(image_path.resolve()),
                "--output-dir",
                str(output_dir.resolve()),
            ]
        )
    finally:
        os.chdir(previous)

    assert result == 0
    assert (output_dir / "scene.jpg").is_file()
