from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .types import Detection


COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 128, 0),
    (0, 190, 255),
    (255, 80, 110),
    (80, 220, 100),
    (180, 100, 255),
    (40, 210, 210),
    (255, 160, 60),
    (90, 160, 255),
    (220, 220, 50),
    (200, 90, 200),
    (80, 200, 170),
    (150, 150, 255),
    (255, 210, 110),
    (180, 180, 180),
)


def draw_detections(
    image: np.ndarray, detections: Sequence[Detection]
) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR image with shape HxWx3")

    output = image.copy()
    height, width = output.shape[:2]
    for detection in detections:
        color = COLORS[detection.class_id % len(COLORS)]
        x1, y1, x2, y2 = detection.box
        left = min(max(int(round(x1)), 0), max(width - 1, 0))
        top = min(max(int(round(y1)), 0), max(height - 1, 0))
        right = min(max(int(round(x2)), 0), max(width - 1, 0))
        bottom = min(max(int(round(y2)), 0), max(height - 1, 0))
        cv2.rectangle(output, (left, top), (right, bottom), color, 2, cv2.LINE_AA)

        caption = f"{detection.label} {detection.score:.2f}"
        text_y = max(top - 7, 14)
        cv2.putText(
            output,
            caption,
            (left, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return output
