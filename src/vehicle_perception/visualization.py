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
    occupied_captions: list[tuple[int, int, int, int]] = []
    for detection in detections:
        color = COLORS[detection.class_id % len(COLORS)]
        x1, y1, x2, y2 = detection.box
        left = min(max(int(round(x1)), 0), max(width - 1, 0))
        top = min(max(int(round(y1)), 0), max(height - 1, 0))
        right = min(max(int(round(x2)), 0), max(width - 1, 0))
        bottom = min(max(int(round(y2)), 0), max(height - 1, 0))
        caption = f"{detection.label} {detection.score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        caption_width = min(text_width + 8, width)
        caption_height = min(text_height + baseline + 7, height)
        text_x = min(max(left, 0), max(width - caption_width, 0))
        preferred_top = max(top - caption_height - 2, 0)

        caption_top = preferred_top
        for offset in range(0, height, max(caption_height, 1)):
            candidate_top = min(preferred_top + offset, max(height - caption_height, 0))
            candidate = (
                text_x,
                candidate_top,
                text_x + caption_width,
                candidate_top + caption_height,
            )
            overlaps = any(
                candidate[0] < existing[2]
                and candidate[2] > existing[0]
                and candidate[1] < existing[3]
                and candidate[3] > existing[1]
                for existing in occupied_captions
            )
            if not overlaps:
                caption_top = candidate_top
                break
        caption_bounds = (
            text_x,
            caption_top,
            text_x + caption_width,
            caption_top + caption_height,
        )
        occupied_captions.append(caption_bounds)

        cv2.rectangle(output, (left, top), (right, bottom), color, 2, cv2.LINE_AA)
        cv2.rectangle(
            output,
            (caption_bounds[0], caption_bounds[1]),
            (caption_bounds[2], caption_bounds[3]),
            color,
            cv2.FILLED,
            cv2.LINE_AA,
        )
        text_color = (15, 25, 40) if sum(color) > 400 else (245, 248, 250)
        cv2.putText(
            output,
            caption,
            (text_x + 4, caption_top + text_height + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            cv2.LINE_AA,
        )
    return output
