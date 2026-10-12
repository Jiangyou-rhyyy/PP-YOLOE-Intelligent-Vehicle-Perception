from __future__ import annotations

import json

from vehicle_perception.io import load_labels, write_competition_json
from vehicle_perception.types import Detection


def test_load_labels_preserves_spaces_and_skips_blank_lines(tmp_path) -> None:
    labels_path = tmp_path / "labels.txt"
    labels_path.write_text("traffic light\n\n  other vehicle  \ncar\n", encoding="utf-8")

    assert load_labels(labels_path) == ["traffic light", "other vehicle", "car"]


def test_write_competition_json_serializes_detections(tmp_path) -> None:
    destination = tmp_path / "nested" / "scene.json"
    detections = [Detection(2, "car", 0.91, (1.5, 2.5, 30.0, 40.0))]

    payload = write_competition_json(
        destination,
        image_name="scene",
        detections=detections,
        timestamp=1_700_000_000,
    )

    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert payload["name"] == "scene"
    assert payload["frames"][0]["timestamp"] == 1_700_000_000
    assert payload["frames"][0]["objects"] == [
        {
            "category": "car",
            "id": 0,
            "attributes": {
                "occluded": "none",
                "truncated": "none",
                "trafficLightColor": "none",
            },
            "box2d": {"x1": 1.5, "y1": 2.5, "x2": 30.0, "y2": 40.0},
            "score": 0.91,
        }
    ]


def test_write_competition_json_supports_empty_detections(tmp_path) -> None:
    destination = tmp_path / "empty.json"

    payload = write_competition_json(
        destination,
        image_name="empty",
        detections=[],
        timestamp=42,
    )

    assert payload["frames"][0]["objects"] == []
    assert json.loads(destination.read_text(encoding="utf-8"))["frames"][0][
        "objects"
    ] == []
