"""Provenance layer: checksums, manifest integrity and verification."""

import hashlib
import logging

import pytest

from climate_impact.provenance import (
    Issue,
    build_entry,
    load_manifest,
    record,
    save_manifest,
    sha256_file,
    to_relative,
    verify,
)

# NIST FIPS 180-2 published test vector: SHA-256("abc")
ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
RAW_FILE = "data/raw/gistemp/GLB.Ts+dSST.csv"


@pytest.fixture
def project(tmp_path):
    """A miniature project root with one downloaded raw file."""
    raw = tmp_path / "data" / "raw" / "gistemp"
    raw.mkdir(parents=True)
    data_file = raw / "GLB.Ts+dSST.csv"
    data_file.write_bytes(b"abc")
    return tmp_path, data_file, tmp_path / "data" / "raw" / "manifest.json"


def make_entry(path, root, **overrides):
    meta = {
        "dataset": "gistemp_v4_global_monthly",
        "source_url": "https://example.org/gistemp.csv",
        "version": "GISTEMP v4",
        "license": "Public domain",
        "retrieval": "http",
    }
    meta.update(overrides)
    return build_entry(path, root=root, **meta)


def test_sha256_matches_nist_test_vector(project):
    _, data_file, _ = project
    assert sha256_file(data_file) == ABC_SHA256


def test_streamed_hash_equals_in_memory_hash_for_large_file(tmp_path):
    payload = bytes(range(256)) * 50_000  # ~12.8 MB, larger than any read buffer
    big = tmp_path / "big.bin"
    big.write_bytes(payload)
    assert sha256_file(big) == hashlib.sha256(payload).hexdigest()


def test_paths_are_relative_and_posix(project):
    root, data_file, _ = project
    assert to_relative(data_file, root) == RAW_FILE


def test_path_outside_project_is_rejected(project, tmp_path_factory):
    root, _, _ = project
    outside = tmp_path_factory.mktemp("elsewhere") / "x.csv"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError, match=r"subpath|does not start with"):
        to_relative(outside, root)


def test_manifest_roundtrip_is_byte_identical_and_lf(project):
    root, data_file, manifest = project
    entry = make_entry(data_file, root)
    save_manifest([entry], manifest)
    first = manifest.read_bytes()
    save_manifest(load_manifest(manifest), manifest)
    assert manifest.read_bytes() == first
    assert load_manifest(manifest) == [entry]
    assert b"\r\n" not in first


def test_record_upserts_without_duplicates(project):
    root, data_file, manifest = project
    record(make_entry(data_file, root), manifest)
    record(make_entry(data_file, root, notes="re-run"), manifest)
    entries = load_manifest(manifest)
    assert len(entries) == 1
    assert entries[0].notes == "re-run"


def test_new_data_vintage_is_warned(project, caplog):
    root, data_file, manifest = project
    record(make_entry(data_file, root), manifest)
    data_file.write_bytes(b"abd")  # provider revised the data
    with caplog.at_level(logging.WARNING):
        previous = record(make_entry(data_file, root), manifest)
    assert previous is not None
    assert previous.sha256 == ABC_SHA256
    assert "New data vintage" in caplog.text


def test_verify_passes_on_untouched_data(project):
    root, data_file, manifest = project
    record(make_entry(data_file, root), manifest)
    assert verify(manifest, raw_dir=root / "data" / "raw", root=root) == []


def test_verify_detects_tampering_unregistered_and_missing(project):
    root, data_file, manifest = project
    raw_dir = root / "data" / "raw"
    record(make_entry(data_file, root), manifest)

    data_file.write_bytes(b"tampered")
    (raw_dir / "stray.csv").write_bytes(b"?")
    issues = verify(manifest, raw_dir=raw_dir, root=root)
    assert Issue(RAW_FILE, "checksum_mismatch") in issues
    assert Issue("data/raw/stray.csv", "unregistered") in issues

    data_file.unlink()
    assert Issue(RAW_FILE, "missing") in verify(manifest, raw_dir=raw_dir, root=root)
