from __future__ import annotations

import numpy as np

import vehicle_perception.visualization as visualization
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


def test_draw_detections_staggers_overlapping_captions(monkeypatch) -> None:
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    detections = [
        Detection(6, "car", 0.91, (20, 30, 100, 90)),
        Detection(6, "car", 0.88, (22, 31, 105, 95)),
    ]
    origins: list[tuple[int, int]] = []
    real_put_text = visualization.cv2.putText

    def recording_put_text(image, text, origin, *args, **kwargs):
        origins.append(origin)
        return real_put_text(image, text, origin, *args, **kwargs)

    monkeypatch.setattr(visualization.cv2, "putText", recording_put_text)

    draw_detections(image, detections)

    assert len(origins) == 2
    assert abs(origins[0][1] - origins[1][1]) >= 14
