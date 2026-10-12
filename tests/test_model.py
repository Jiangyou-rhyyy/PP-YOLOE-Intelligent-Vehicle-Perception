from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import vehicle_perception.model as model_module
from vehicle_perception.model import OpenVINODetector


class FakePort:
    def __init__(self, name: str) -> None:
        self.any_name = name


class FakeDimension:
    is_static = True

    def get_length(self) -> int:
        return 640


class DynamicBatchPort(FakePort):
    @property
    def shape(self) -> tuple[int, ...]:
        raise RuntimeError("to_shape was called on a dynamic shape")

    @property
    def partial_shape(self) -> list[FakeDimension]:
        return [FakeDimension(), FakeDimension(), FakeDimension(), FakeDimension()]


class FakeCompiledModel:
    def __init__(self, outputs: list[np.ndarray]) -> None:
        self.outputs = outputs
        self.last_input: np.ndarray | None = None

    def input(self, index: int) -> FakePort:
        assert index == 0
        return FakePort("image")

    def __call__(self, inputs: list[np.ndarray]) -> dict[FakePort, np.ndarray]:
        self.last_input = inputs[0]
        return {
            FakePort(f"output-{index}"): output
            for index, output in enumerate(self.outputs)
        }


class DynamicInputCompiledModel(FakeCompiledModel):
    def input(self, index: int) -> DynamicBatchPort:
        assert index == 0
        return DynamicBatchPort("image")


class FakeCore:
    def __init__(self, outputs: list[np.ndarray]) -> None:
        self.outputs = outputs
        self.read_path: Path | None = None
        self.device: str | None = None
        self.compiled = FakeCompiledModel(outputs)

    def read_model(self, model: str) -> object:
        self.read_path = Path(model)
        return object()

    def compile_model(self, model: object, device_name: str) -> FakeCompiledModel:
        self.device = device_name
        return self.compiled


def make_model_pair(tmp_path: Path) -> Path:
    xml_path = tmp_path / "model.xml"
    xml_path.write_text("<net />", encoding="utf-8")
    xml_path.with_suffix(".bin").write_bytes(b"weights")
    return xml_path


def test_detector_selects_outputs_by_shape_not_dictionary_order(
    tmp_path, monkeypatch
) -> None:
    xml_path = make_model_pair(tmp_path)
    boxes = np.array([[[10, 20, 30, 40]]], dtype=np.float32)
    scores = np.zeros((1, 14, 1), dtype=np.float32)
    scores[0, 6, 0] = 0.9
    core = FakeCore([scores, boxes])
    monkeypatch.setattr(model_module, "_create_core", lambda: core)
    detector = OpenVINODetector(
        xml_path,
        [f"class-{index}" for index in range(13)],
        device="CPU",
        extra_channel_policy="label_unknown",
    )

    detections = detector.predict(np.zeros((100, 200, 3), dtype=np.uint8))

    assert core.device == "CPU"
    assert core.read_path == xml_path.resolve()
    assert core.compiled.last_input is not None
    assert core.compiled.last_input.shape == (1, 3, 640, 640)
    assert [item.label for item in detections] == ["class-6"]


def test_detector_reads_static_spatial_size_from_dynamic_batch_input(
    tmp_path, monkeypatch
) -> None:
    xml_path = make_model_pair(tmp_path)
    core = FakeCore([])
    core.compiled = DynamicInputCompiledModel([])
    monkeypatch.setattr(model_module, "_create_core", lambda: core)

    detector = OpenVINODetector(xml_path, ["car"])

    assert detector.input_size == 640


def test_detector_works_from_a_different_working_directory(
    tmp_path, monkeypatch
) -> None:
    xml_path = make_model_pair(tmp_path)
    boxes = np.zeros((1, 1, 4), dtype=np.float32)
    scores = np.zeros((1, 1, 1), dtype=np.float32)
    core = FakeCore([boxes, scores])
    monkeypatch.setattr(model_module, "_create_core", lambda: core)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    previous = Path.cwd()
    try:
        os.chdir(elsewhere)
        detector = OpenVINODetector(xml_path.resolve(), ["car"])
        assert detector.predict(np.zeros((10, 10, 3), dtype=np.uint8)) == []
    finally:
        os.chdir(previous)


@pytest.mark.parametrize("missing", ["xml", "bin"])
def test_detector_requires_complete_openvino_model_pair(
    tmp_path, monkeypatch, missing
) -> None:
    xml_path = tmp_path / "model.xml"
    if missing == "bin":
        xml_path.write_text("<net />", encoding="utf-8")
    monkeypatch.setattr(model_module, "_create_core", lambda: FakeCore([]))

    with pytest.raises(FileNotFoundError, match=f"model.{missing}"):
        OpenVINODetector(xml_path, ["car"])


def test_detector_rejects_unverified_extra_score_channel(tmp_path, monkeypatch) -> None:
    xml_path = make_model_pair(tmp_path)
    boxes = np.zeros((1, 1, 4), dtype=np.float32)
    scores = np.zeros((1, 14, 1), dtype=np.float32)
    core = FakeCore([boxes, scores])
    monkeypatch.setattr(model_module, "_create_core", lambda: core)
    detector = OpenVINODetector(xml_path, [f"class-{i}" for i in range(13)])

    with pytest.raises(ValueError, match="14 score channels.*13 labels"):
        detector.predict(np.zeros((10, 10, 3), dtype=np.uint8))


def test_detector_rejects_ambiguous_output_shapes(tmp_path, monkeypatch) -> None:
    xml_path = make_model_pair(tmp_path)
    core = FakeCore(
        [
            np.zeros((1, 1, 4), dtype=np.float32),
            np.zeros((1, 1, 4), dtype=np.float32),
        ]
    )
    monkeypatch.setattr(model_module, "_create_core", lambda: core)
    detector = OpenVINODetector(xml_path, ["car"])

    with pytest.raises(ValueError, match="identify one boxes output"):
        detector.predict(np.zeros((10, 10, 3), dtype=np.uint8))
