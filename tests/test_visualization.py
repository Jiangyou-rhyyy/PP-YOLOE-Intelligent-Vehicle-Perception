from __future__ import annotations

import numpy as np

from vehicle_perception.types import Detection
from vehicle_perception.visualization import draw_detections


def test_draw_detections_returns_annotated_copy_without_mutating_input() -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    original = image.copy()
    detections = [
        Detection(
            class_id=6,
            label="car",
            score=0.9123,
            box=(10.0, 20.0, 90.0, 60.0),
        )
    ]

    annotated = draw_detections(image, detections)

    np.testing.assert_array_equal(image, original)
    assert annotated is not image
    assert np.any(annotated != original)


def test_draw_detections_handles_unknown_class_and_empty_results() -> None:
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    unknown = Detection(99, "unknown", 0.75, (5.0, 10.0, 50.0, 40.0))

    annotated = draw_detections(image, [unknown])
    empty = draw_detections(image, [])

    assert np.any(annotated != image)
    np.testing.assert_array_equal(empty, image)
    assert empty is not image
