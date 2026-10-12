from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .io import load_labels
from .postprocess import ExtraChannelPolicy, postprocess_detections
from .preprocess import preprocess_image
from .types import Detection


def _create_core() -> Any:
    try:
        from openvino import Core
    except ImportError as exc:  # pragma: no cover - depends on optional runtime install
        raise RuntimeError(
            "OpenVINO is not installed. Install the project runtime requirements first."
        ) from exc
    return Core()


def _port_name(port: Any, fallback: str) -> str:
    try:
        return str(port.any_name)
    except (AttributeError, RuntimeError):
        return fallback


def _input_size(port: Any, default: int = 640) -> int:
    try:
        shape = port.shape
        if shape:
            return int(shape[-1])
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    try:
        dimension = port.partial_shape[-1]
        if bool(dimension.is_static):
            return int(dimension.get_length())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return default


class OpenVINODetector:
    def __init__(
        self,
        model_path: str | Path,
        labels: Sequence[str],
        device: str = "CPU",
        *,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.35,
        extra_channel_policy: ExtraChannelPolicy = "error",
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"model.xml not found: {self.model_path}")
        weights_path = self.model_path.with_suffix(".bin")
        if not weights_path.is_file():
            raise FileNotFoundError(f"model.bin not found: {weights_path}")
        if not labels:
            raise ValueError("At least one label is required")

        self.labels = list(labels)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.extra_channel_policy = extra_channel_policy
        self.last_output_shapes: dict[str, list[int]] = {}

        core = _create_core()
        model = core.read_model(model=str(self.model_path))
        self.compiled_model = core.compile_model(model, device_name=device)
        self.input_port = self.compiled_model.input(0)
        self.input_size = _input_size(self.input_port)

    @staticmethod
    def _select_outputs(
        inference_result: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]]]:
        if hasattr(inference_result, "items"):
            named_outputs = [
                (_port_name(port, f"output-{index}"), np.asarray(value))
                for index, (port, value) in enumerate(inference_result.items())
            ]
        else:
            named_outputs = [
                (f"output-{index}", np.asarray(value))
                for index, value in enumerate(inference_result)
            ]

        boxes_candidates = [
            (name, value)
            for name, value in named_outputs
            if value.ndim == 3 and value.shape[-1] == 4
        ]
        if len(boxes_candidates) != 1:
            shapes = {name: list(value.shape) for name, value in named_outputs}
            raise ValueError(
                f"Could not identify one boxes output by trailing dimension 4: {shapes}"
            )
        boxes_name, boxes = boxes_candidates[0]
        candidate_count = boxes.shape[1]

        scores_candidates = [
            (name, value)
            for name, value in named_outputs
            if name != boxes_name
            and value.ndim == 3
            and value.shape[-1] == candidate_count
        ]
        if len(scores_candidates) != 1:
            shapes = {name: list(value.shape) for name, value in named_outputs}
            raise ValueError(
                "Could not identify one scores output matching the boxes candidate count: "
                f"{shapes}"
            )
        _, scores = scores_candidates[0]
        shapes = {name: list(value.shape) for name, value in named_outputs}
        return boxes, scores, shapes

    def predict(self, image: np.ndarray) -> list[Detection]:
        tensor, transform = preprocess_image(image, self.input_size)
        result = self.compiled_model([tensor])
        boxes, scores, self.last_output_shapes = self._select_outputs(result)
        return postprocess_detections(
            boxes,
            scores,
            self.labels,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            extra_channel_policy=self.extra_channel_policy,
            transform=transform,
        )


def smoke_test(
    model_path: Path, image_path: Path, labels_path: Path
) -> dict[str, object]:
    image = cv2.imread(str(image_path.expanduser().resolve()))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    labels = load_labels(labels_path)
    detector = OpenVINODetector(
        model_path,
        labels,
        device="CPU",
        extra_channel_policy="label_unknown",
    )
    detections = detector.predict(image)
    return {
        "device": detector.device,
        "input_name": _port_name(detector.input_port, "input-0"),
        "input_size": detector.input_size,
        "output_shapes": detector.last_output_shapes,
        "detection_count": len(detections),
        "detections": [
            {
                "class_id": detection.class_id,
                "label": detection.label,
                "score": detection.score,
                "box": list(detection.box),
            }
            for detection in detections
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenVINO PP-YOLOE+ model tools")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--image", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.smoke_test:
        raise SystemExit("Specify --smoke-test")
    missing = [name for name in ("model", "labels", "image") if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"Missing required smoke-test arguments: {', '.join(missing)}")
    result = smoke_test(args.model, args.image, args.labels)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
