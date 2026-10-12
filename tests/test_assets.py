from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def test_source_images_are_readable() -> None:
    for name in ("day.jpg", "night.jpg"):
        image = cv2.imread(str(ASSETS / "source" / name))
        assert image is not None and image.size > 0, name


def test_demo_images_match_sources_and_contain_annotations() -> None:
    for source_name, demo_name in (
        ("day.jpg", "demo-day.jpg"),
        ("night.jpg", "demo-night.jpg"),
    ):
        source = cv2.imread(str(ASSETS / "source" / source_name))
        demo = cv2.imread(str(ASSETS / demo_name))
        assert source is not None and demo is not None
        assert demo.shape == source.shape
        assert np.mean(np.abs(demo.astype(np.int16) - source.astype(np.int16))) > 0.5


def test_demo_gif_has_multiple_frames_and_is_compact() -> None:
    gif_path = ASSETS / "demo.gif"
    assert gif_path.stat().st_size < 8 * 1024 * 1024
    with Image.open(gif_path) as image:
        assert image.n_frames >= 2
        assert image.info["duration"] >= 1000


def test_benchmark_svg_contains_verified_values() -> None:
    svg = (ASSETS / "benchmark.svg").read_text(encoding="utf-8")
    for expected in (
        "PP-YOLOE+_s",
        "PP-YOLOE+_m",
        "PP-YOLOE+_l",
        "40.23%",
        "17.59 FPS",
        "2.76 MB",
        "45.8%",
        "14.97 FPS",
        "28.7 MB",
        "47.9%",
        "10.88 FPS",
        "196 MB",
    ):
        assert expected in svg
