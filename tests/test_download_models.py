from __future__ import annotations

import hashlib
import io
import zipfile

import pytest


def archive_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()


def test_release_asset_url_is_stable_and_named() -> None:
    from scripts.download_models import release_asset_url

    assert release_asset_url("owner/repo", "v1.0.0-models", "bundle.zip") == (
        "https://github.com/owner/repo/releases/download/v1.0.0-models/bundle.zip"
    )


def test_download_model_bundle_verifies_and_extracts_expected_files(
    tmp_path, monkeypatch
) -> None:
    from scripts import download_models

    bundle = archive_bytes(
        {"model.xml": b"graph", "model.bin": b"weights", "model.mapping": b"mapping"}
    )
    checksum = f"{hashlib.sha256(bundle).hexdigest()}  ppyoloe-openvino-model.zip\n".encode()
    responses = [Response(bundle), Response(checksum)]
    monkeypatch.setattr(download_models, "urlopen", lambda *args, **kwargs: responses.pop(0))

    destination = download_models.download_model_bundle(
        tmp_path / "model", repo="owner/repo", tag="v1.0.0-models"
    )

    assert destination == (tmp_path / "model").resolve()
    assert (destination / "model.xml").read_bytes() == b"graph"
    assert (destination / "model.bin").read_bytes() == b"weights"
    assert (destination / "model.mapping").read_bytes() == b"mapping"


def test_download_model_bundle_skips_existing_complete_model(tmp_path, monkeypatch) -> None:
    from scripts import download_models

    destination = tmp_path / "model"
    destination.mkdir()
    (destination / "model.xml").write_bytes(b"graph")
    (destination / "model.bin").write_bytes(b"weights")
    monkeypatch.setattr(download_models, "urlopen", lambda *args, **kwargs: pytest.fail("network used"))

    assert download_models.download_model_bundle(destination) == destination.resolve()


@pytest.mark.parametrize(
    "bundle, checksum",
    [
        (archive_bytes({"model.xml": b"graph", "model.bin": b"weights"}), b"0" * 64),
        (archive_bytes({"../escape.txt": b"bad", "model.xml": b"graph", "model.bin": b"weights"}), None),
    ],
)
def test_invalid_or_unsafe_download_leaves_no_partial_model_files(
    tmp_path, monkeypatch, bundle, checksum
) -> None:
    from scripts import download_models

    if checksum is None:
        checksum = f"{hashlib.sha256(bundle).hexdigest()}  ppyoloe-openvino-model.zip\n".encode()
    responses = [Response(bundle), Response(checksum)]
    monkeypatch.setattr(download_models, "urlopen", lambda *args, **kwargs: responses.pop(0))
    destination = tmp_path / "model"

    with pytest.raises(RuntimeError):
        download_models.download_model_bundle(destination)

    assert not (destination / "model.xml").exists()
    assert not (destination / "model.bin").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_network_failure_leaves_no_partial_model_files(tmp_path, monkeypatch) -> None:
    from scripts import download_models

    def fail(*args, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(download_models, "urlopen", fail)
    destination = tmp_path / "model"

    with pytest.raises(RuntimeError, match="Unable to download"):
        download_models.download_model_bundle(destination)

    assert not (destination / "model.xml").exists()
    assert not (destination / "model.bin").exists()
