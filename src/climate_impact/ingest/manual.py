"""Register a manually downloaded raw file in the provenance manifest (Phase 2, Step 2.4e).

Some sources cannot be scripted: EM-DAT needs a logged-in export, and some bulk files block
non-browser clients. The file is downloaded by hand into data/raw/, then this command checks
it and records its SHA-256, source, access date and export settings (retrieval: manual).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from climate_impact.config import PROJECT_ROOT, get_path, load_datasets
from climate_impact.ingest.http import HTML_MARKERS, DownloadError
from climate_impact.ingest.source_check import MIN_PLAUSIBLE_BYTES
from climate_impact.provenance import build_entry, load_manifest, record, sha256_file, to_relative

logger = logging.getLogger(__name__)


def register(
    name: str,
    meta: dict[str, Any],
    path: Path,
    *,
    source_url: str,
    notes: str,
    raw_dir: Path,
    manifest: Path,
    root: Path = PROJECT_ROOT,
    force: bool = False,
) -> str:
    """Validate a hand-downloaded raw file and record it; return its manifest path."""
    path = path.resolve()
    if not path.is_file():
        raise DownloadError(f"{path} does not exist")
    if not path.is_relative_to(raw_dir.resolve()):
        raise DownloadError(f"{path} must be inside {raw_dir} (the raw data layer)")
    size = path.stat().st_size
    if size < MIN_PLAUSIBLE_BYTES:
        raise DownloadError(f"{path.name}: only {size} bytes; not a plausible data file")
    with path.open("rb") as f:
        if f.read(64).lstrip().lower().startswith(HTML_MARKERS):
            raise DownloadError(f"{path.name} is an HTML page, not a data file")

    rel = to_relative(path, root)
    previous = next((e for e in load_manifest(manifest) if e.key == (name, rel)), None)
    if previous is not None:
        if previous.sha256 == sha256_file(path):
            logger.info("%s already registered with the same checksum", rel)
            return rel
        if not force:
            raise DownloadError(f"{rel} differs from its registered checksum; use --force")

    entry = build_entry(
        path,
        dataset=name,
        source_url=source_url,
        version=str(meta.get("version", "")),
        license=str(meta.get("license", "")),
        retrieval="manual",
        notes=notes,
        root=root,
    )
    record(entry, manifest)
    logger.info("registered %s (%d bytes, sha256 %s...)", rel, size, entry.sha256[:12])
    return rel


def main(argv: list[str] | None = None) -> int:
    """Run ``python -m climate_impact.ingest.manual <dataset> <path> --source-url URL``."""
    parser = argparse.ArgumentParser(description="Register a manually downloaded raw file.")
    parser.add_argument("dataset", help="name from config/datasets.yaml")
    parser.add_argument("path", type=Path, help="file inside data/raw/")
    parser.add_argument("--source-url", required=True, help="page or link it came from")
    parser.add_argument("--notes", default="", help="export filters, release date, etc.")
    parser.add_argument("--force", action="store_true", help="register a new vintage")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    catalogue = load_datasets()
    if args.dataset not in catalogue:
        logger.error("unknown dataset %r (see config/datasets.yaml)", args.dataset)
        return 1
    try:
        register(
            args.dataset,
            catalogue[args.dataset],
            args.path,
            source_url=args.source_url,
            notes=args.notes,
            raw_dir=get_path("raw"),
            manifest=get_path("manifest"),
            force=args.force,
        )
    except DownloadError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
