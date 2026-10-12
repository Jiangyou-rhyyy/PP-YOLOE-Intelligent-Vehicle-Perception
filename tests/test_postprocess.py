from __future__ import annotations

import numpy as np
import pytest

from vehicle_perception.postprocess import postprocess_detections
from vehicle_perception.types import ImageTransform


def test_postprocess_filters_confidence_and_applies_class_aware_nms() -> None:
    boxes = np.array(
        [[[10, 10, 100, 100], [12, 12, 98, 98], [10, 10, 100, 100]]],
        dtype=np.float32,
    )
    scores = np.array(
        [[[0.90, 0.80, 0.10], [0.05, 0.10, 0.85]]],
        dtype=np.float32,
    )

    detections = postprocess_detections(
        boxes,
        scores,
        ["car", "bus"],
        confidence_threshold=0.5,
        iou_threshold=0.35,
    )

    assert [item.label for item in detections] == ["car", "bus"]
    assert [item.score for item in detections] == pytest.approx([0.90, 0.85])


def test_postprocess_clips_boxes_to_original_image() -> None:
    boxes = np.array([[[-10, -20, 700, 700]]], dtype=np.float32)
    scores = np.array([[[0.9]]], dtype=np.float32)
    transform = ImageTransform(100, 200, 200, 640)

    detections = postprocess_detections(
        boxes,
        scores,
        ["car"],
        transform=transform,
    )

    assert detections[0].box == pytest.approx((0.0, 0.0, 200.0, 100.0))


def test_postprocess_returns_empty_list_when_no_score_reaches_threshold() -> None:
    boxes = np.array([[[10, 10, 20, 20]]], dtype=np.float32)
    scores = np.array([[[0.49]]], dtype=np.float32)

    assert postprocess_detections(boxes, scores, ["car"]) == []


def test_postprocess_accepts_thirteen_channels_for_thirteen_labels() -> None:
    boxes = np.array([[[10, 10, 20, 20]]], dtype=np.float32)
    scores = np.zeros((1, 13, 1), dtype=np.float32)
    scores[0, 12, 0] = 0.9
    labels = [f"class-{index}" for index in range(13)]

    detections = postprocess_detections(boxes, scores, labels)

    assert detections[0].class_id == 12
    assert detections[0].label == "class-12"


def test_postprocess_rejects_unverified_extra_score_channel() -> None:
    boxes = np.array([[[10, 10, 20, 20]]], dtype=np.float32)
    scores = np.zeros((1, 14, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="14 score channels.*13 labels"):
        postprocess_detections(boxes, scores, [f"class-{i}" for i in range(13)])


def test_postprocess_can_explicitly_drop_verified_final_channel() -> None:
    boxes = np.array([[[10, 10, 20, 20]]], dtype=np.float32)
    scores = np.zeros((1, 14, 1), dtype=np.float32)
    scores[0, 0, 0] = 0.8
    scores[0, 13, 0] = 0.99

    detections = postprocess_detections(
        boxes,
        scores,
        [f"class-{i}" for i in range(13)],
        extra_channel_policy="drop_last",
    )

    assert len(detections) == 1
    assert detections[0].class_id == 0
    assert detections[0].label == "class-0"


def test_postprocess_can_label_verified_extra_channel_as_unknown() -> None:
    boxes = np.array([[[10, 10, 20, 20]]], dtype=np.float32)
    scores = np.zeros((1, 14, 1), dtype=np.float32)
    scores[0, 13, 0] = 0.9

    detections = postprocess_detections(
        boxes,
        scores,
        [f"class-{i}" for i in range(13)],
        extra_channel_policy="label_unknown",
    )

    assert detections[0].class_id == 13
    assert detections[0].label == "unknown"


def test_postprocess_rejects_malformed_output_shapes() -> None:
    with pytest.raises(ValueError, match="boxes shape"):
        postprocess_detections(
            np.zeros((1, 4), dtype=np.float32),
            np.zeros((1, 1, 1), dtype=np.float32),
            ["car"],
        )
