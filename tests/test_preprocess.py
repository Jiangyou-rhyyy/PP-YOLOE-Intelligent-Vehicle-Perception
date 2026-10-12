from __future__ import annotations

import numpy as np
import pytest

from vehicle_perception.preprocess import preprocess_image


def test_preprocess_converts_bgr_to_normalized_rgb_nchw() -> None:
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    image[0, 0] = [0, 128, 255]

    tensor, transform = preprocess_image(image, input_size=640)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0
    np.testing.assert_allclose(
        tensor[0, :, 0, 0],
        np.array([1.0, 128 / 255.0, 0.0], dtype=np.float32),
    )
    assert transform.original_height == 2
    assert transform.original_width == 4
    assert transform.padded_size == 4
    assert transform.input_size == 640


@pytest.mark.parametrize(
    ("shape", "model_box", "expected"),
    [
        ((2, 4, 3), (0.0, 0.0, 640.0, 320.0), (0.0, 0.0, 4.0, 2.0)),
        ((4, 2, 3), (0.0, 0.0, 320.0, 640.0), (0.0, 0.0, 2.0, 4.0)),
    ],
)
def test_transform_maps_model_box_to_landscape_or_portrait_image(
    shape: tuple[int, int, int],
    model_box: tuple[float, float, float, float],
    expected: tuple[float, float, float, float],
) -> None:
    image = np.zeros(shape, dtype=np.uint8)

    _, transform = preprocess_image(image, input_size=640)

    assert transform.to_original_box(model_box) == pytest.approx(expected)


def test_preprocess_rejects_empty_image() -> None:
    with pytest.raises(ValueError, match="non-empty BGR image"):
        preprocess_image(np.empty((0, 0, 3), dtype=np.uint8))
