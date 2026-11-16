"""ERA5 monthly means via the Copernicus Climate Data Store (Phase 2, Step 2.4d).

The CDS queue can hold a request for hours, so years are grouped into multi-year requests,
several requests are kept in the queue at once, and request IDs are remembered in
.cache/era5_jobs.json so that a re-run reattaches to waiting requests instead of submitting
duplicates. The CDS delivers each request as a ZIP holding one NetCDF file per variable
type (t2m instantaneous, tp accumulated); the ZIP is validated and stored unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tempfile
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import requests
import xarray as xr

from climate_impact.config import PROJECT_ROOT, get_path, load_datasets
from climate_impact.ingest.http import DownloadError, _check_existing, target_path
from climate_impact.provenance import build_entry, record, to_relative

logger = logging.getLogger(__name__)

DATASET = "era5_monthly_single_levels"
STATE_FILE = PROJECT_ROOT / ".cache" / "era5_jobs.json"
NETCDF_MAGIC = (b"\x89HDF\r\n\x1a\n", b"CDF")
EXPECTED_UNITS = {"t2m": "K", "tp": "m"}
ERA5_GRID = (721, 1440)
CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class Chunk:
    """A contiguous block of years fetched in one CDS request."""

    first: int
    last: int

    @property
    def label(self) -> str:
        """Identify the chunk, e.g. '1960_1965'."""
        return f"{self.first}_{self.last}"

    @property
    def years(self) -> list[int]:
        """Return all years in the chunk."""
        return list(range(self.first, self.last + 1))


def make_chunks(first: int, last: int, size: int) -> list[Chunk]:
    """Split [first, last] into consecutive chunks of at most ``size`` years."""
    return [Chunk(y, min(y + size - 1, last)) for y in range(first, last + 1, size)]


def chunk_request(meta: dict[str, Any], chunk: Chunk) -> dict[str, Any]:
    """Build the CDS request for one chunk from the catalogue entry (no mutation)."""
    request = dict(meta["request"])
    request["year"] = [str(y) for y in chunk.years]
    return request


class _CdsApi(Protocol):
    def submit(self, dataset: str, request: dict[str, Any]) -> str: ...

    def status(self, job_id: str) -> str: ...

    def download(self, job_id: str, target: Path) -> None: ...


def _hrefs(obj: object) -> Iterator[str]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "href":
                yield str(value)
            else:
                yield from _hrefs(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _hrefs(value)


class RestCdsApi:
    """Minimal client for the CDS retrieve API: submit, poll, download."""

    def __init__(self, url: str, key: str) -> None:
        self.base = url.rstrip("/") + "/retrieve/v1"
        self.session = requests.Session()
        self.session.headers["PRIVATE-TOKEN"] = key

    @classmethod
    def from_rc(cls, rc: Path | None = None) -> RestCdsApi:
        """Build a client from ~/.cdsapirc (url and key lines)."""
        text = (rc or Path.home() / ".cdsapirc").read_text(encoding="utf-8")
        cfg = dict(re.findall(r"^(url|key):\s*(\S+)", text, re.MULTILINE))
        return cls(cfg["url"], cfg["key"])

    def submit(self, dataset: str, request: dict[str, Any]) -> str:
        """Queue a request and return its job ID."""
        r = self.session.post(
            f"{self.base}/processes/{dataset}/execution", json={"inputs": request}, timeout=60
        )
        r.raise_for_status()
        return str(r.json()["jobID"])

    def status(self, job_id: str) -> str:
        """Return the job status: accepted, running, successful, failed or dismissed."""
        r = self.session.get(f"{self.base}/jobs/{job_id}", timeout=60)
        if r.status_code == 404:
            return "missing"
        r.raise_for_status()
        return str(r.json()["status"])

    def download(self, job_id: str, target: Path) -> None:
        """Stream the job's result to ``target`` (the storage link is fetched without the token)."""
        r = self.session.get(f"{self.base}/jobs/{job_id}/results", timeout=60)
        r.raise_for_status()
        links = list(_hrefs(r.json()))
        if not links:
            raise DownloadError(f"job {job_id}: no download link in results")
        with requests.get(links[0], stream=True, timeout=(30, 600)) as resp:
            resp.raise_for_status()
            with target.open("wb") as f:
                for block in resp.iter_content(CHUNK_BYTES):
                    f.write(block)


def validate_era5_zip(
    path: Path, years: list[int], grid: tuple[int, int] | None = ERA5_GRID
) -> None:
    """Raise DownloadError unless the ZIP holds NetCDF with t2m (K) and tp (m) for every month.

    Timestamps are normalized to the calendar month before comparison: the CDS stamps
    accumulated fields (tp) at 06:00 and instantaneous fields (t2m) at 00:00.
    """
    if not zipfile.is_zipfile(path):
        raise DownloadError(f"{path.name}: not a ZIP archive")
    expected = set(pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-01", freq="MS"))
    found: dict[str, set[pd.Timestamp]] = {}
    with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory() as tmp:
        for name in archive.namelist():
            member = Path(archive.extract(name, tmp))
            with member.open("rb") as f:
                if not f.read(8).startswith(NETCDF_MAGIC):
                    raise DownloadError(f"{path.name}:{name}: not a NetCDF file")
            with xr.open_dataset(member) as ds:
                shape = (ds.sizes.get("latitude"), ds.sizes.get("longitude"))
                if grid is not None and shape != grid:
                    raise DownloadError(f"{path.name}:{name}: unexpected grid {shape}")
                stamps = pd.DatetimeIndex(ds["valid_time"].values)
                months = set(stamps.to_period("M").to_timestamp())
                for var, unit in EXPECTED_UNITS.items():
                    if var in ds:
                        units = ds[var].attrs.get("units")
                        if units != unit:
                            raise DownloadError(f"{path.name}: {var} units {units!r}, not {unit!r}")
                        found[var] = months
    for var in EXPECTED_UNITS:
        if var not in found:
            raise DownloadError(f"{path.name}: missing variable {var}")
        if found[var] != expected:
            raise DownloadError(
                f"{path.name}: {var} covers {len(found[var])} of {len(expected)} months"
            )


def _target(meta: dict[str, Any], chunk: Chunk, raw_dir: Path) -> Path:
    raw_path = str(meta["raw_path"]).format(first=chunk.first, last=chunk.last)
    return target_path({"raw_path": raw_path}, raw_dir)


def _store(
    meta: dict[str, Any],
    chunk: Chunk,
    job_id: str,
    *,
    api: _CdsApi,
    raw_dir: Path,
    manifest: Path,
    root: Path,
    grid: tuple[int, int] | None,
) -> str:
    target = _target(meta, chunk, raw_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    try:
        api.download(job_id, part)
        validate_era5_zip(part, chunk.years, grid=grid)
        part.replace(target)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    entry = build_entry(
        target,
        dataset=DATASET,
        source_url=f"cds://{meta['cds_dataset']}",
        version=str(meta.get("version", "ERA5")),
        license=str(meta.get("license", "")),
        retrieval="cdsapi",
        notes="ZIP as delivered: one NetCDF per stepType (t2m instantaneous, tp accumulated)",
        request={**chunk_request(meta, chunk), "job_id": job_id},
        root=root,
    )
    record(entry, manifest)
    logger.info("%s: stored %s (%d bytes)", chunk.label, entry.file, target.stat().st_size)
    return entry.file


def run_once(
    meta: dict[str, Any],
    chunks: list[Chunk],
    *,
    api: _CdsApi,
    state: dict[str, str],
    raw_dir: Path,
    manifest: Path,
    root: Path = PROJECT_ROOT,
    max_active: int = 4,
    grid: tuple[int, int] | None = ERA5_GRID,
) -> dict[str, int]:
    """Make one pass: store finished requests, then submit new ones up to ``max_active``.

    ``state`` maps chunk label -> CDS job ID and is updated in place.
    """
    counts = dict.fromkeys(("done", "stored", "waiting", "submitted", "failed"), 0)
    pending: list[Chunk] = []
    for chunk in chunks:
        target = _target(meta, chunk, raw_dir)
        if target.exists():
            _check_existing(DATASET, target, to_relative(target, root), manifest)
            state.pop(chunk.label, None)
            counts["done"] += 1
            continue
        job_id = state.get(chunk.label)
        if job_id is None:
            pending.append(chunk)
            continue
        status = api.status(job_id)
        if status == "successful":
            try:
                _store(
                    meta,
                    chunk,
                    job_id,
                    api=api,
                    raw_dir=raw_dir,
                    manifest=manifest,
                    root=root,
                    grid=grid,
                )
            except DownloadError as exc:  # keep the job ID so the same result can be inspected
                logger.error("%s: result did not validate: %s", chunk.label, exc)
                counts["failed"] += 1
                continue
            state.pop(chunk.label, None)
            counts["stored"] += 1
        elif status in {"accepted", "running"}:
            counts["waiting"] += 1
        else:
            logger.error("%s: request %s ended as %r", chunk.label, job_id, status)
            state.pop(chunk.label, None)
            counts["failed"] += 1

    if counts["failed"]:
        return counts  # never queue new work in a pass that saw a failure; investigate first

    active = counts["waiting"]
    for chunk in pending:
        if active >= max_active:
            break
        state[chunk.label] = api.submit(str(meta["cds_dataset"]), chunk_request(meta, chunk))
        logger.info("%s: submitted request %s", chunk.label, state[chunk.label])
        active += 1
        counts["submitted"] += 1
    return counts


def load_state(path: Path = STATE_FILE) -> dict[str, str]:
    """Read remembered job IDs (chunk label -> job ID)."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.items()}


def save_state(state: dict[str, str], path: Path = STATE_FILE) -> None:
    """Write remembered job IDs atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    """Run ``python -m climate_impact.ingest.era5 [--once] [--poll-minutes N]``."""
    parser = argparse.ArgumentParser(description="ERA5 monthly means into data/raw/era5.")
    parser.add_argument("--once", action="store_true", help="one pass, then exit")
    parser.add_argument("--poll-minutes", type=float, default=10.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    meta = load_datasets()[DATASET]
    first, last = meta["years"]
    chunks = make_chunks(int(first), int(last), int(meta["years_per_request"]))
    api = RestCdsApi.from_rc()
    state = load_state()
    while True:
        try:
            counts = run_once(
                meta,
                chunks,
                api=api,
                state=state,
                raw_dir=get_path("raw"),
                manifest=get_path("manifest"),
                max_active=int(meta["max_active_requests"]),
            )
        finally:
            save_state(state)
        logger.info("pass: %s", counts)
        if counts["failed"]:
            logger.error("Stopping: a request failed or a result did not validate (see above).")
            return 1
        if counts["done"] + counts["stored"] == len(chunks):
            logger.info("All %d chunks stored.", len(chunks))
            return 0
        if args.once:
            return 0
        time.sleep(args.poll_minutes * 60)


if __name__ == "__main__":
    sys.exit(main())
