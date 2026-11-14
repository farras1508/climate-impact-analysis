"""ERA5 monthly means via the Copernicus Climate Data Store API (Phase 2, Step 2.4d).

One request per year (small requests queue faster and fail independently). Each file is
validated as a 12-month ERA5 product (t2m in K, tp in m) before it enters data/raw/, then
recorded in the manifest. Needs a CDS account, the accepted ERA5 licence and ~/.cdsapirc.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Protocol

import cdsapi
import xarray as xr

from climate_impact.config import PROJECT_ROOT, get_path, load_datasets
from climate_impact.ingest.http import DownloadError, DownloadResult, _check_existing, target_path
from climate_impact.provenance import build_entry, record, to_relative

logger = logging.getLogger(__name__)

DATASET = "era5_monthly_single_levels"
NETCDF_MAGIC = (b"\x89HDF\r\n\x1a\n", b"CDF")
EXPECTED_UNITS = {"t2m": "K", "tp": "m"}
MIN_BYTES = 1_000_000


class _Client(Protocol):
    def retrieve(self, name: str, request: dict[str, Any], target: str) -> object: ...


def year_request(meta: dict[str, Any], year: int) -> dict[str, Any]:
    """Build the CDS request for one year from the catalogue entry (no mutation)."""
    request = dict(meta["request"])
    request["year"] = [str(year)]
    return request


def validate_era5(path: Path, min_bytes: int = MIN_BYTES) -> None:
    """Raise DownloadError unless path is a 12-month ERA5 file with t2m (K) and tp (m)."""
    size = path.stat().st_size
    if size < min_bytes:
        raise DownloadError(f"{path.name}: implausibly small ({size} bytes)")
    with path.open("rb") as f:
        if not f.read(8).startswith(NETCDF_MAGIC):
            raise DownloadError(f"{path.name}: not a NetCDF file")
    with xr.open_dataset(path) as ds:
        time_dim = "valid_time" if "valid_time" in ds.dims else "time"
        n_months = ds.sizes.get(time_dim)
        if n_months != 12:
            raise DownloadError(f"{path.name}: expected 12 months, got {n_months}")
        for var, unit in EXPECTED_UNITS.items():
            if var not in ds:
                raise DownloadError(f"{path.name}: missing variable {var}")
            found = ds[var].attrs.get("units")
            if found != unit:
                raise DownloadError(f"{path.name}: {var} units {found!r}, expected {unit!r}")


def download_year(
    year: int,
    meta: dict[str, Any],
    *,
    client: _Client,
    raw_dir: Path,
    manifest: Path,
    root: Path = PROJECT_ROOT,
    min_bytes: int = MIN_BYTES,
) -> DownloadResult:
    """Retrieve, validate and register one year of ERA5 monthly means."""
    target = target_path({**meta, "raw_path": str(meta["raw_path"]).format(year=year)}, raw_dir)
    rel = to_relative(target, root)
    if target.exists():
        skipped = _check_existing(DATASET, target, rel, manifest)
        if skipped is not None:
            return skipped

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    request = year_request(meta, year)
    try:
        client.retrieve(str(meta["cds_dataset"]), request, str(part))
        validate_era5(part, min_bytes=min_bytes)  # file is closed again before the rename
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
        request=request,
        root=root,
    )
    record(entry, manifest)
    logger.info("%d: stored %s (%d bytes)", year, rel, target.stat().st_size)
    return DownloadResult(DATASET, rel, "downloaded", entry.sha256)


def main(argv: list[str] | None = None) -> int:
    """Download ERA5 by year: ``python -m climate_impact.ingest.era5 [--years FIRST LAST]``."""
    parser = argparse.ArgumentParser(description="ERA5 monthly means into data/raw/era5.")
    parser.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    meta = load_datasets()[DATASET]
    first, last = args.years or meta["years"]
    client = cdsapi.Client()
    failures = 0
    for year in range(first, last + 1):
        try:
            download_year(
                year, meta, client=client, raw_dir=get_path("raw"), manifest=get_path("manifest")
            )
        except Exception as exc:  # cdsapi raises generic exceptions; log and continue
            failures += 1
            logger.error("%d: %s", year, exc)
    logger.info("Done: years %d-%d, %d failed.", first, last, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
