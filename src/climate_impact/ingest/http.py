"""Provenance-aware HTTP downloader (Phase 2, Step 2.3).

Each download is streamed to a temporary file, hashed while streaming, checked for
plausibility, atomically moved into data/raw/ and recorded in the manifest. Re-runs are
idempotent: a file whose manifest checksum still matches is skipped without any request.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from climate_impact.config import PROJECT_ROOT, get_path, load_datasets
from climate_impact.ingest.source_check import MIN_PLAUSIBLE_BYTES, USER_AGENT, resolve_url
from climate_impact.provenance import build_entry, load_manifest, record, sha256_file, to_relative

logger = logging.getLogger(__name__)

CHUNK_BYTES = 1 << 20
TIMEOUT_S = (30, 300)  # (connect, read) seconds
HTML_MARKERS = (b"<!doctype html", b"<html")


class DownloadError(RuntimeError):
    """A download failed a provenance or plausibility check; nothing was stored."""


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of one download request."""

    dataset: str
    path: str
    action: str  # one of: downloaded, skipped
    sha256: str


def make_session(total_retries: int = 4) -> requests.Session:
    """Return a session that retries transient failures with exponential backoff."""
    retry = Retry(
        total=total_retries,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = USER_AGENT
    return session


def target_path(meta: dict[str, Any], raw_dir: Path) -> Path:
    """Resolve the dataset's raw_path inside raw_dir, refusing paths that escape it."""
    target = (raw_dir / str(meta["raw_path"])).resolve()
    if not target.is_relative_to(raw_dir.resolve()):
        raise ValueError(f"raw_path escapes the raw data directory: {meta['raw_path']}")
    return target


def _stream_to_part(response: requests.Response, part: Path) -> str:
    digest = hashlib.sha256()
    head = b""
    with part.open("wb") as f:
        for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
            if not head:
                head = chunk[:64]
            digest.update(chunk)
            f.write(chunk)
    if head.lstrip().lower().startswith(HTML_MARKERS):
        raise DownloadError("server returned an HTML page instead of a data file")
    return digest.hexdigest()


def _check_existing(name: str, target: Path, rel: str, manifest: Path) -> DownloadResult | None:
    entry = next((e for e in load_manifest(manifest) if e.key == (name, rel)), None)
    if entry is None:
        raise DownloadError(f"{rel} exists with unknown provenance; use --force to replace it")
    if sha256_file(target) != entry.sha256:
        raise DownloadError(f"{rel} no longer matches its manifest checksum; investigate first")
    logger.info("%s: checksum matches manifest, skipping download", name)
    return DownloadResult(name, rel, "skipped", entry.sha256)


def download(
    name: str,
    meta: dict[str, Any],
    *,
    session: requests.Session,
    raw_dir: Path,
    manifest: Path,
    root: Path = PROJECT_ROOT,
    force: bool = False,
) -> DownloadResult:
    """Download one HTTP dataset into the raw layer and record its provenance."""
    url = resolve_url(meta)
    if url is None:
        raise ValueError(f"{name}: not an HTTP dataset with a direct URL")
    target = target_path(meta, raw_dir)
    rel = to_relative(target, root)
    if target.exists() and not force:
        skipped = _check_existing(name, target, rel, manifest)
        if skipped is not None:
            return skipped

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    try:
        with session.get(
            url, stream=True, timeout=TIMEOUT_S, headers={"Accept-Encoding": "identity"}
        ) as response:
            if response.status_code != 200:
                raise DownloadError(f"{name}: HTTP {response.status_code} from {url}")
            streamed_sha = _stream_to_part(response, part)
            final_url = str(response.url)
            last_modified = response.headers.get("Last-Modified", "")
            expected = response.headers.get("Content-Length", "")
        size = part.stat().st_size
        if expected.isdigit() and int(expected) != size:
            raise DownloadError(f"{name}: truncated download ({size} of {expected} bytes)")
        if size < MIN_PLAUSIBLE_BYTES:
            raise DownloadError(f"{name}: only {size} bytes; refusing to store an implausible file")
        if sha256_file(part) != streamed_sha:
            raise DownloadError(f"{name}: checksum differs between stream and disk")
        part.replace(target)
    except BaseException:
        part.unlink(missing_ok=True)
        raise

    entry = build_entry(
        target,
        dataset=name,
        source_url=final_url,
        version=str(meta.get("version", "")),
        license=str(meta.get("license", "")),
        retrieval="http",
        notes=f"Last-Modified: {last_modified}" if last_modified else "",
        request={"requested_url": url},
        root=root,
    )
    record(entry, manifest)
    logger.info("%s: stored %s (%d bytes, sha256 %s...)", name, rel, size, entry.sha256[:12])
    return DownloadResult(name, rel, "downloaded", entry.sha256)


def main(argv: list[str] | None = None) -> int:
    """Download datasets: ``python -m climate_impact.ingest.http <name>... | all [--force]``."""
    parser = argparse.ArgumentParser(description="Provenance-aware download into data/raw.")
    parser.add_argument("datasets", nargs="+", help="names from config/datasets.yaml, or 'all'")
    parser.add_argument("--force", action="store_true", help="replace an existing raw file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    catalogue = load_datasets()
    if args.datasets == ["all"]:
        names = sorted(n for n, m in catalogue.items() if m.get("access") == "http" and "url" in m)
    else:
        names = args.datasets
    failures = 0
    with make_session() as session:
        for name in names:
            try:
                download(
                    name,
                    catalogue[name],
                    session=session,
                    raw_dir=get_path("raw"),
                    manifest=get_path("manifest"),
                    force=args.force,
                )
            except (DownloadError, ValueError, KeyError, requests.RequestException) as exc:
                failures += 1
                logger.error("%s: %s", name, exc)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
