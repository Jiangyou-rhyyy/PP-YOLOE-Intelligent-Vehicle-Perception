from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from .types import Detection


def load_labels(path: str | Path) -> list[str]:
    labels_path = Path(path).expanduser().resolve()
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise ValueError(f"Labels file is empty: {labels_path}")
    return labels


def write_competition_json(
    path: str | Path,
    image_name: str,
    detections: Sequence[Detection],
    timestamp: int,
) -> dict[str, object]:
    objects = []
    for object_id, detection in enumerate(detections):
        x1, y1, x2, y2 = detection.box
        objects.append(
            {
                "category": detection.label,
                "id": object_id,
                "attributes": {
                    "occluded": "none",
                    "truncated": "none",
                    "trafficLightColor": "none",
                },
                "box2d": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                },
                "score": float(detection.score),
            }
        )

    payload: dict[str, object] = {
        "name": image_name,
        "frames": [{"timestamp": int(timestamp), "objects": objects}],
        "attributes": {"weather": "none", "scene": "none", "timeofday": "none"},
    }
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload
