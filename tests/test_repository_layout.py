from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_SIZE = 20 * 1024 * 1024
IGNORED_PARTS = {".git", ".pytest_cache", ".superpowers", ".venv", "__pycache__"}


def publishable_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not IGNORED_PARTS.intersection(path.parts)
    ]


def test_no_publishable_file_exceeds_project_limit() -> None:
    oversized = {
        path.relative_to(ROOT).as_posix(): path.stat().st_size
        for path in publishable_files()
        if path.stat().st_size > MAX_TRACKED_SIZE
    }

    assert oversized == {}


def test_competition_docx_is_excluded() -> None:
    docx_files = [
        path.relative_to(ROOT).as_posix()
        for path in publishable_files()
        if path.suffix.lower() == ".docx"
    ]

    assert docx_files == []


def test_required_project_files_exist() -> None:
    required = [
        ".gitignore",
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "docs/project-report-zh.pdf",
        "docs/legacy/environment.yaml",
        "docs/legacy/environment.txt",
        "docs/legacy/requirements-training.txt",
        "model/infer_cfg.yml",
        "model/labels.txt",
        "model/README.md",
    ]

    missing = [relative for relative in required if not (ROOT / relative).is_file()]
    assert missing == []


def test_model_readme_lists_required_release_files() -> None:
    manifest = (ROOT / "model" / "README.md").read_text(encoding="utf-8")

    assert "model.xml" in manifest
    assert "model.bin" in manifest
    assert "SHA-256" in manifest
