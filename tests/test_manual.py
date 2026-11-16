"""Manual registration tests (offline)."""

import logging

import pytest

from climate_impact.ingest.http import DownloadError
from climate_impact.ingest.manual import register
from climate_impact.provenance import load_manifest

META = {"license": "CC BY 4.0", "version": "v1"}
PAYLOAD = b"PK\x03\x04" + b"x" * 5000  # starts like a zip file, > 1 KB


@pytest.fixture
def layout(tmp_path):
    raw = tmp_path / "data" / "raw"
    (raw / "wdi").mkdir(parents=True)
    return tmp_path, raw, raw / "manifest.json"


def put(raw, data=PAYLOAD, name="wdi/WDI_CSV.zip"):
    path = raw / name
    path.write_bytes(data)
    return path


def reg(layout, path, **kwargs):
    root, raw, manifest = layout
    return register(
        "wdi",
        META,
        path,
        source_url="https://example.org/WDI_CSV.zip",
        notes="browser download",
        raw_dir=raw,
        manifest=manifest,
        root=root,
        **kwargs,
    )


def test_registers_file_with_manual_provenance(layout):
    _, raw, manifest = layout
    assert reg(layout, put(raw)) == "data/raw/wdi/WDI_CSV.zip"
    [entry] = load_manifest(manifest)
    assert entry.retrieval == "manual"
    assert entry.notes == "browser download"
    assert entry.license == "CC BY 4.0"


def test_registering_the_same_file_twice_is_idempotent(layout):
    _, raw, manifest = layout
    path = put(raw)
    reg(layout, path)
    reg(layout, path)
    assert len(load_manifest(manifest)) == 1


def test_changed_file_needs_force_and_logs_a_new_vintage(layout, caplog):
    _, raw, _ = layout
    path = put(raw)
    reg(layout, path)
    path.write_bytes(PAYLOAD + b"revised")
    with pytest.raises(DownloadError, match="use --force"):
        reg(layout, path)
    with caplog.at_level(logging.WARNING):
        reg(layout, path, force=True)
    assert "New data vintage" in caplog.text


def test_html_page_is_refused(layout):
    _, raw, _ = layout
    with pytest.raises(DownloadError, match="HTML"):
        reg(layout, put(raw, b"<!DOCTYPE html>" + b"x" * 5000))


def test_tiny_file_is_refused(layout):
    _, raw, _ = layout
    with pytest.raises(DownloadError, match="only 10 bytes"):
        reg(layout, put(raw, b"x" * 10))


def test_file_outside_the_raw_layer_is_refused(layout, tmp_path):
    outside = tmp_path / "elsewhere.zip"
    outside.write_bytes(PAYLOAD)
    with pytest.raises(DownloadError, match="must be inside"):
        reg(layout, outside)


def test_missing_file_is_refused(layout):
    _, raw, _ = layout
    with pytest.raises(DownloadError, match="does not exist"):
        reg(layout, raw / "wdi" / "nothing.zip")
