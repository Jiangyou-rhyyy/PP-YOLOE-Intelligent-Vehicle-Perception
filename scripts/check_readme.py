from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
REQUIRED = (
    "## Overview",
    "## Core Features",
    "## Demo",
    "## Quick Start",
    "## Project Structure",
    "## Results",
    "## Dataset & Acknowledgements",
    "40.23%",
    "17.59 FPS",
    "47.9%",
    "https://www.vis.xyz/bdd100k/",
    "https://github.com/PaddlePaddle/PaddleDetection",
    "https://arxiv.org/abs/2203.16250",
    "https://docs.openvino.ai/",
)
PROHIBITED = ("D:\\", "C:\\", "TODO", "TBD", "1960930003@qq.com")
ASSETS = ("assets/demo.gif", "assets/demo-day.jpg", "assets/demo-night.jpg", "assets/benchmark.svg")


def main() -> int:
    text = README.read_text(encoding="utf-8")
    errors = [f"missing required text: {item}" for item in REQUIRED if item not in text]
    errors.extend(f"prohibited text: {item}" for item in PROHIBITED if item in text)
    errors.extend(f"missing referenced asset: {item}" for item in ASSETS if not (ROOT / item).is_file())
    if errors:
        print("README validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("README validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
