"""Data provenance: SHA-256 checksums and the raw-data manifest.

Every file in ``data/raw`` gets a manifest entry recording where it came from,
when, under which licence, and its SHA-256 digest. The manifest is committed
to git (DEC-016); the data files are not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from climate_impact.config import PROJECT_ROOT, get_path

logger = logging.getLogger(__name__)

IGNORED_RAW_FILES = frozenset({".gitkeep", "manifest.json"})


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, streamed in constant memory."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def to_relative(path: Path, root: Path = PROJECT_ROOT) -> str:
    """Return ``path`` relative to ``root`` as a POSIX string.

    POSIX separators keep manifest keys identical on Windows and Linux (CI).
    Raises ``ValueError`` if ``path`` lies outside ``root``.
    """
    return path.resolve().relative_to(root.resolve()).as_posix()


@dataclass(frozen=True)
class ManifestEntry:
    """Provenance record for one raw file."""

    dataset: str
    file: str
    sha256: str
    size_bytes: int
    source_url: str
    version: str
    access_date: str
    license: str
    retrieval: str
    notes: str = ""
    request: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        """Identify the entry: exactly one record per (dataset, file)."""
        return (self.dataset, self.file)


def build_entry(
    path: Path,
    *,
    dataset: str,
    source_url: str,
    version: str,
    license: str,
    retrieval: str,
    notes: str = "",
    request: dict[str, Any] | None = None,
    root: Path = PROJECT_ROOT,
) -> ManifestEntry:
    """Hash a downloaded file and build its manifest entry (access date in UTC)."""
    return ManifestEntry(
        dataset=dataset,
        file=to_relative(path, root),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        source_url=source_url,
        version=version,
        access_date=datetime.now(UTC).date().isoformat(),
        license=license,
        retrieval=retrieval,
        notes=notes,
        request=request or {},
    )


def load_manifest(manifest: Path | None = None) -> list[ManifestEntry]:
    """Read the manifest; a missing file means no data has been acquired yet."""
    manifest = manifest or get_path("manifest")
    if not manifest.exists():
        return []
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{manifest} must contain a JSON list of entries.")
    return [ManifestEntry(**item) for item in raw]


def save_manifest(entries: list[ManifestEntry], manifest: Path | None = None) -> None:
    """Write the manifest deterministically and atomically.

    Entries and keys are sorted so the file diffs cleanly in git. The payload is
    written to a temporary file that then replaces the original, so an
    interrupted run can never leave a half-written manifest.
    """
    manifest = manifest or get_path("manifest")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(entries, key=lambda e: e.key)
    payload = json.dumps([asdict(e) for e in ordered], indent=2, sort_keys=True, ensure_ascii=False)
    fd, tmp_name = tempfile.mkstemp(dir=manifest.parent, suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload + "\n")
        tmp.replace(manifest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def record(entry: ManifestEntry, manifest: Path | None = None) -> ManifestEntry | None:
    """Insert or update an entry and return the entry it replaced, if any.

    A changed checksum for the same file means the provider published a new
    vintage (e.g. GISTEMP revises past months as late station data arrive).
    This is logged as a warning and must be recorded in docs/decision_log.md.
    """
    entries = load_manifest(manifest)
    previous = next((e for e in entries if e.key == entry.key), None)
    if previous is not None and previous.sha256 != entry.sha256:
        logger.warning(
            "New data vintage for %s (%s): sha256 %s... -> %s...",
            entry.dataset,
            entry.file,
            previous.sha256[:12],
            entry.sha256[:12],
        )
    save_manifest([*(e for e in entries if e.key != entry.key), entry], manifest)
    return previous


@dataclass(frozen=True)
class Issue:
    """A provenance problem found during verification."""

    file: str
    problem: str  # one of: missing, checksum_mismatch, unregistered


def verify(
    manifest: Path | None = None,
    raw_dir: Path | None = None,
    root: Path = PROJECT_ROOT,
) -> list[Issue]:
    """Re-hash every registered file and flag raw files with no manifest entry."""
    entries = load_manifest(manifest)
    issues: list[Issue] = []
    for entry in entries:
        target = root / entry.file
        if not target.is_file():
            issues.append(Issue(entry.file, "missing"))
        elif sha256_file(target) != entry.sha256:
            issues.append(Issue(entry.file, "checksum_mismatch"))

    registered = {entry.file for entry in entries}
    for path in sorted((raw_dir or get_path("raw")).rglob("*")):
        if path.is_file() and path.name not in IGNORED_RAW_FILES and path.suffix != ".tmp":
            rel = to_relative(path, root)
            if rel not in registered:
                issues.append(Issue(rel, "unregistered"))
    return issues


def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point ``python -m climate_impact.provenance verify``."""
    parser = argparse.ArgumentParser(description="Verify raw data against the manifest.")
    parser.add_argument("command", choices=["verify"])
    parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    issues = verify()
    for issue in issues:
        logger.error("%s: %s", issue.problem, issue.file)
    logger.info("%d registered file(s) checked, %d issue(s).", len(load_manifest()), len(issues))
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
