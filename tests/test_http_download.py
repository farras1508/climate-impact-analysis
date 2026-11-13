"""Downloader tests run offline against a fake HTTP session."""

import hashlib
import logging

import pytest

from climate_impact.ingest.http import DownloadError, download, target_path
from climate_impact.provenance import load_manifest

PAYLOAD = b"year,anomaly\n" + b"2000,0.40\n" * 200  # > 1 KB, plausible CSV
META = {
    "access": "http",
    "url": "https://example.org/series.csv",
    "raw_path": "demo/series.csv",
    "license": "CC BY 4.0",
    "version": "v1",
}


class FakeResponse:
    def __init__(self, body, status_code=200, headers=None, url="https://example.org/series.csv"):
        self.body = body
        self.status_code = status_code
        self.headers = {"Content-Length": str(len(body)), **(headers or {})}
        self.url = url

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        return self.response


@pytest.fixture
def layout(tmp_path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    return tmp_path, raw, raw / "manifest.json"


def run(layout, session, meta=META, **kwargs):
    root, raw, manifest = layout
    return download(
        "demo", meta, session=session, raw_dir=raw, manifest=manifest, root=root, **kwargs
    )


def test_download_stores_file_and_records_provenance(layout):
    _, raw, manifest = layout
    result = run(layout, FakeSession(FakeResponse(PAYLOAD)))
    assert result.action == "downloaded"
    assert (raw / "demo" / "series.csv").read_bytes() == PAYLOAD
    [entry] = load_manifest(manifest)
    assert entry.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    assert entry.file == "data/raw/demo/series.csv"
    assert entry.retrieval == "http"
    assert entry.license == "CC BY 4.0"


def test_rerun_is_skipped_without_any_request(layout):
    run(layout, FakeSession(FakeResponse(PAYLOAD)))
    session = FakeSession(FakeResponse(PAYLOAD))
    assert run(layout, session).action == "skipped"
    assert session.calls == 0


def test_html_error_page_is_rejected_and_nothing_is_written(layout):
    _, raw, manifest = layout
    page = b"<!DOCTYPE html><html>" + b"x" * 2000
    with pytest.raises(DownloadError, match="HTML"):
        run(layout, FakeSession(FakeResponse(page)))
    assert not any(raw.rglob("*.csv*"))
    assert load_manifest(manifest) == []


def test_truncated_download_is_rejected_and_cleaned_up(layout):
    _, raw, _ = layout
    response = FakeResponse(PAYLOAD, headers={"Content-Length": str(len(PAYLOAD) + 100)})
    with pytest.raises(DownloadError, match="truncated"):
        run(layout, FakeSession(response))
    assert not any(raw.rglob("*.csv*"))


def test_implausibly_small_file_is_rejected(layout):
    with pytest.raises(DownloadError, match="implausible"):
        run(layout, FakeSession(FakeResponse(b"year,anomaly\n")))


def test_http_error_status_is_rejected(layout):
    with pytest.raises(DownloadError, match="HTTP 404"):
        run(layout, FakeSession(FakeResponse(PAYLOAD, status_code=404)))


def test_existing_file_without_manifest_entry_is_refused(layout):
    _, raw, _ = layout
    stray = raw / "demo" / "series.csv"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(PAYLOAD)
    with pytest.raises(DownloadError, match="unknown provenance"):
        run(layout, FakeSession(FakeResponse(PAYLOAD)))


def test_tampered_file_is_not_silently_replaced(layout):
    _, raw, _ = layout
    run(layout, FakeSession(FakeResponse(PAYLOAD)))
    (raw / "demo" / "series.csv").write_bytes(PAYLOAD + b"tamper")
    with pytest.raises(DownloadError, match="no longer matches"):
        run(layout, FakeSession(FakeResponse(PAYLOAD)))


def test_forced_refresh_records_a_new_vintage(layout, caplog):
    _, _, manifest = layout
    run(layout, FakeSession(FakeResponse(PAYLOAD)))
    revised = PAYLOAD.replace(b"0.40", b"0.41")
    with caplog.at_level(logging.WARNING):
        run(layout, FakeSession(FakeResponse(revised)), force=True)
    [entry] = load_manifest(manifest)
    assert entry.sha256 == hashlib.sha256(revised).hexdigest()
    assert "New data vintage" in caplog.text


def test_raw_path_cannot_escape_the_raw_directory(layout):
    _, raw, _ = layout
    with pytest.raises(ValueError, match="escapes"):
        target_path({**META, "raw_path": "../../outside.csv"}, raw)
