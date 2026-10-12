from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from .types import Detection, ImageTransform


ExtraChannelPolicy = Literal["error", "drop_last", "label_unknown"]


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, float(box[2] - box[0])) * max(
        0.0, float(box[3] - box[1])
    )
    other_areas = np.maximum(0.0, others[:, 2] - others[:, 0]) * np.maximum(
        0.0, others[:, 3] - others[:, 1]
    )
    union = box_area + other_areas - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        selected = int(order[0])
        keep.append(selected)
        if order.size == 1:
            break
        remaining = order[1:]
        order = remaining[_iou(boxes[selected], boxes[remaining]) <= iou_threshold]
    return keep


def postprocess_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: Sequence[str],
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.35,
    extra_channel_policy: ExtraChannelPolicy = "error",
    transform: ImageTransform | None = None,
) -> list[Detection]:
    boxes_array = np.asarray(boxes, dtype=np.float32)
    scores_array = np.asarray(scores, dtype=np.float32)
    if boxes_array.ndim != 3 or boxes_array.shape[0] != 1 or boxes_array.shape[2] != 4:
        raise ValueError(
            f"Expected boxes shape [1, candidates, 4], got {boxes_array.shape}"
        )
    if scores_array.ndim != 3 or scores_array.shape[0] != 1:
        raise ValueError(
            f"Expected scores shape [1, classes, candidates], got {scores_array.shape}"
        )
    if scores_array.shape[2] != boxes_array.shape[1]:
        raise ValueError("Boxes and scores must contain the same candidate count")
    if not labels:
        raise ValueError("At least one label is required")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")

    channel_count = scores_array.shape[1]
    resolved_labels = list(labels)
    if channel_count != len(resolved_labels):
        if channel_count == len(resolved_labels) + 1 and extra_channel_policy == "drop_last":
            scores_array = scores_array[:, :-1, :]
        elif (
            channel_count == len(resolved_labels) + 1
            and extra_channel_policy == "label_unknown"
        ):
            resolved_labels.append("unknown")
        else:
            raise ValueError(
                f"Model exposes {channel_count} score channels but {len(labels)} labels "
                f"were supplied; policy={extra_channel_policy!r}"
            )

    candidate_boxes = boxes_array[0].copy()
    x1 = np.minimum(candidate_boxes[:, 0], candidate_boxes[:, 2])
    y1 = np.minimum(candidate_boxes[:, 1], candidate_boxes[:, 3])
    x2 = np.maximum(candidate_boxes[:, 0], candidate_boxes[:, 2])
    y2 = np.maximum(candidate_boxes[:, 1], candidate_boxes[:, 3])
    candidate_boxes = np.column_stack((x1, y1, x2, y2))

    candidate_scores = scores_array[0]
    class_ids = np.argmax(candidate_scores, axis=0)
    confidences = np.max(candidate_scores, axis=0)
    eligible = np.flatnonzero(confidences >= confidence_threshold)
    if eligible.size == 0:
        return []

    kept_candidates: list[int] = []
    for class_id in sorted(set(int(class_ids[index]) for index in eligible)):
        class_candidates = eligible[class_ids[eligible] == class_id]
        local_keep = _nms(
            candidate_boxes[class_candidates],
            confidences[class_candidates],
            iou_threshold,
        )
        kept_candidates.extend(int(class_candidates[index]) for index in local_keep)

    detections: list[Detection] = []
    for index in sorted(kept_candidates, key=lambda item: float(-confidences[item])):
        class_id = int(class_ids[index])
        raw_box = tuple(float(value) for value in candidate_boxes[index])
        output_box = transform.to_original_box(raw_box) if transform else raw_box
        detections.append(
            Detection(
                class_id=class_id,
                label=resolved_labels[class_id],
                score=float(confidences[index]),
                box=output_box,
            )
        )
    return detections
