from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageTransform:
    original_height: int
    original_width: int
    padded_size: int
    input_size: int

    def to_original_box(
        self, box: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        scale = self.padded_size / self.input_size
        x1, y1, x2, y2 = (float(value) * scale for value in box)
        return (
            min(max(x1, 0.0), float(self.original_width)),
            min(max(y1, 0.0), float(self.original_height)),
            min(max(x2, 0.0), float(self.original_width)),
            min(max(y2, 0.0), float(self.original_height)),
        )


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    score: float
    box: tuple[float, float, float, float]
