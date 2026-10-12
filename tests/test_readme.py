from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_has_required_title_sections_assets_and_sources() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert readme.startswith(
        "# Intelligent Vehicle Environment Perception and Real-Time Object Detection System Based on PP-YOLOE+"
    )
    for heading in (
        "## Overview",
        "## Core Features",
        "## Demo",
        "## Quick Start",
        "## Project Structure",
        "## Results",
        "## Dataset & Acknowledgements",
    ):
        assert heading in readme
    for asset in (
        "assets/demo.gif",
        "assets/demo-day.jpg",
        "assets/demo-night.jpg",
        "assets/benchmark.svg",
    ):
        assert asset in readme
    for source in (
        "https://www.vis.xyz/bdd100k/",
        "https://github.com/PaddlePaddle/PaddleDetection",
        "https://arxiv.org/abs/2203.16250",
        "https://docs.openvino.ai/",
    ):
        assert source in readme


def test_readme_has_complete_quick_start_and_verified_benchmarks() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for required in (
        "git clone https://github.com/Jiangyou-rhyyy/PP-YOLOE-Intelligent-Vehicle-Perception.git",
        "pip install -r requirements.txt",
        "python scripts/download_models.py",
        "python scripts/detect_images.py",
        "python scripts/detect_webcam.py",
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
        assert required in readme


def test_readme_has_no_machine_locality_or_placeholder_copy() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for prohibited in ("D:\\", "C:\\", "TODO", "TBD", "1960930003@qq.com"):
        assert prohibited not in readme
