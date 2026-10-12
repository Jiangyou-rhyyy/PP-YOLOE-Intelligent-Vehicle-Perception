from __future__ import annotations

import cv2
import numpy as np

from .types import ImageTransform


def preprocess_image(
    image: np.ndarray, input_size: int = 640
) -> tuple[np.ndarray, ImageTransform]:
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] == 0
        or image.shape[1] == 0
    ):
        raise ValueError("Expected a non-empty BGR image with shape HxWx3")
    if input_size <= 0:
        raise ValueError("input_size must be positive")

    height, width = image.shape[:2]
    padded_size = max(height, width)
    padded = np.zeros((padded_size, padded_size, 3), dtype=np.uint8)
    padded[:height, :width] = image

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
    tensor = np.transpose(resized, (2, 0, 1))[np.newaxis].astype(np.float32)
    tensor /= 255.0

    return np.ascontiguousarray(tensor), ImageTransform(
        original_height=height,
        original_width=width,
        padded_size=padded_size,
        input_size=input_size,
    )
