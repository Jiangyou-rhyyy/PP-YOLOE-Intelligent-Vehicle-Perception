from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = "Jiangyou-rhyyy/PP-YOLOE-Intelligent-Vehicle-Perception"
DEFAULT_TAG = "v1.0.0-models"
ARCHIVE_NAME = "ppyoloe-openvino-model.zip"
CHECKSUM_NAME = f"{ARCHIVE_NAME}.sha256"
EXPECTED_FILES = ("model.xml", "model.bin")
OPTIONAL_FILES = ("model.mapping",)


def release_asset_url(repository: str, tag: str, asset_name: str) -> str:
    return f"https://github.com/{repository}/releases/download/{tag}/{asset_name}"


def _download(url: str) -> bytes:
    try:
        with urlopen(url, timeout=60) as response:
            return response.read()
    except OSError as exc:
        raise RuntimeError(f"Unable to download {url}: {exc}") from exc


def _expected_sha256(checksum: bytes) -> str:
    try:
        digest = checksum.decode("utf-8").strip().split()[0].lower()
    except (IndexError, UnicodeDecodeError) as exc:
        raise RuntimeError("Release checksum file is malformed") from exc
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RuntimeError("Release checksum file does not contain a SHA-256 digest")
    return digest


def _safe_members(archive: zipfile.ZipFile) -> list[str]:
    names = archive.namelist()
    allowed = set(EXPECTED_FILES + OPTIONAL_FILES)
    if any(
        Path(name).is_absolute()
        or ".." in Path(name).parts
        or Path(name).name != name
        or name not in allowed
        for name in names
    ):
        raise RuntimeError("Release archive contains an unsafe or unexpected file")
    if not set(EXPECTED_FILES).issubset(names):
        raise RuntimeError("Release archive is missing model.xml or model.bin")
    return names


def download_model_bundle(
    destination: str | Path,
    *,
    repo: str = DEFAULT_REPOSITORY,
    tag: str = DEFAULT_TAG,
) -> Path:
    destination_path = Path(destination).expanduser().resolve()
    if all((destination_path / name).is_file() for name in EXPECTED_FILES):
        return destination_path

    archive_url = release_asset_url(repo, tag, ARCHIVE_NAME)
    checksum_url = release_asset_url(repo, tag, CHECKSUM_NAME)
    archive_bytes = _download(archive_url)
    expected_digest = _expected_sha256(_download(checksum_url))
    actual_digest = hashlib.sha256(archive_bytes).hexdigest()
    if actual_digest != expected_digest:
        raise RuntimeError(
            "Downloaded model archive failed SHA-256 verification "
            f"({actual_digest} != {expected_digest})"
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="model-download-", dir=destination_path.parent))
    try:
        try:
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                for name in _safe_members(archive):
                    (staging / name).write_bytes(archive.read(name))
        except zipfile.BadZipFile as exc:
            raise RuntimeError("Downloaded model archive is not a valid ZIP file") from exc

        destination_path.mkdir(parents=True, exist_ok=True)
        for name in EXPECTED_FILES + OPTIONAL_FILES:
            source = staging / name
            if source.is_file():
                source.replace(destination_path / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return destination_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and verify OpenVINO model release assets")
    parser.add_argument("--destination", type=Path, default=ROOT / "model")
    parser.add_argument("--repo", default=DEFAULT_REPOSITORY)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    args = parser.parse_args(argv)
    try:
        destination = download_model_bundle(
            args.destination, repo=args.repo, tag=args.tag
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Verified OpenVINO model is available in {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
