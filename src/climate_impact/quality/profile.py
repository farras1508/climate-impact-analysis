"""Step 3.1: first-pass, domain-aware profiling of every raw dataset (DEC-021).

For each table: rows, columns, cells, missing cells, the period covered (all rows and
rows with a measured value), number of entities, sentinel codes replaced at read, the
units and baseline *declared* in ``config/datasets.yaml``, and a per-column profile.
ERA5 is inspected per ZIP chunk without loading a full grid into memory.

Outputs (deterministic: re-running on the same data vintage gives identical files):

* ``reports/profiling/profile.json``: machine-readable, used by Steps 3.2-3.6;
* ``reports/profiling/<dataset>.md``: one human-readable profile per dataset;
* ``reports/profiling/inventory.md``: one line per table.

Licence safeguard: value frequencies are listed only for columns with at most
``MAX_LISTED_CATEGORIES`` distinct values, so no record-level text (EM-DAT event names,
locations) can enter a committed report.

Usage: ``python -m climate_impact.quality.profile [DATASET ...] [--skip-era5]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from climate_impact.config import PROJECT_ROOT, get_path, load_datasets
from climate_impact.ingest.readers import READERS, RawTable, read_dataset
from climate_impact.provenance import ManifestEntry, load_manifest, sha256_file

logger = logging.getLogger(__name__)

ERA5 = "era5_monthly_single_levels"
MAX_LISTED_CATEGORIES = 50
TOP_VALUES = 8
DECLARED_FIELDS = (
    "provider",
    "version",
    "native_units",
    "native_baseline",
    "missing_sentinel",
    "license",
)


# ---------------------------------------------------------------- tabular profiles
def _num(x: object) -> float | int | None:
    """Convert numpy/pandas scalars to JSON-safe Python numbers (NaN -> None)."""
    if x is None or (isinstance(x, float | np.floating) and np.isnan(x)):
        return None
    if isinstance(x, np.integer | int):
        return int(x)
    return float(x)  # type: ignore[arg-type]


def _stamp(x: object) -> str | None:
    if x is None or pd.isna(x):  # type: ignore[call-overload]
        return None
    return str(pd.Timestamp(x).date())  # type: ignore[arg-type]


def profile_column(values: pd.Series) -> dict[str, Any]:
    """Describe one column: type, missingness, cardinality, and range or frequencies."""
    n = len(values)
    n_missing = int(values.isna().sum())
    present = values.dropna()
    out: dict[str, Any] = {
        "dtype": str(values.dtype),
        "n_missing": n_missing,
        "pct_missing": round(100 * n_missing / n, 2) if n else None,
        "n_unique": int(present.nunique()),
    }
    if pd.api.types.is_bool_dtype(values):
        out["kind"] = "boolean"
        out["n_true"] = int(present.astype(bool).sum())
    elif pd.api.types.is_numeric_dtype(values):
        out["kind"] = "numeric"
        q = present.quantile([0.0, 0.01, 0.5, 0.99, 1.0]) if len(present) else None
        stats = {
            "min": q.iloc[0] if q is not None else None,
            "p01": q.iloc[1] if q is not None else None,
            "median": q.iloc[2] if q is not None else None,
            "p99": q.iloc[3] if q is not None else None,
            "max": q.iloc[4] if q is not None else None,
            "mean": present.mean() if len(present) else None,
            "std": present.std() if len(present) > 1 else None,
        }
        out.update({k: _num(v) for k, v in stats.items()})
        out["n_negative"] = int((present < 0).sum())
        out["n_zero"] = int((present == 0).sum())
    elif pd.api.types.is_datetime64_any_dtype(values):
        out["kind"] = "datetime"
        out["min"] = _stamp(present.min()) if len(present) else None
        out["max"] = _stamp(present.max()) if len(present) else None
    else:
        out["kind"] = "text"
        lengths = present.astype(str).str.len()
        out["min_length"] = _num(lengths.min()) if len(present) else None
        out["max_length"] = _num(lengths.max()) if len(present) else None
        if out["n_unique"] <= MAX_LISTED_CATEGORIES:
            counts = present.astype(str).value_counts().head(TOP_VALUES)
            out["top_values"] = {str(k): int(v) for k, v in counts.items()}
    return out


def _period_range(series: pd.Series, freq: str | None) -> dict[str, Any]:
    """First/last period and the number of distinct periods."""
    present = series.dropna()
    if present.empty:
        return {"first": None, "last": None, "n_periods": 0}
    if freq == "M":
        first, last = _stamp(present.min()), _stamp(present.max())
        return {
            "first": first[:7] if first else None,
            "last": last[:7] if last else None,
            "n_periods": int(present.nunique()),
        }
    return {
        "first": _num(present.min()),
        "last": _num(present.max()),
        "n_periods": int(present.nunique()),
    }


def profile_table(table: RawTable) -> dict[str, Any]:
    """Profile one table read by :mod:`climate_impact.ingest.readers`."""
    frame = table.frame
    n_cells = int(frame.size)
    n_missing = int(frame.isna().sum().sum())
    out: dict[str, Any] = {
        "table": table.table,
        "rows": len(frame),
        "columns": frame.shape[1],
        "cells": n_cells,
        "missing_cells": n_missing,
        "pct_missing_cells": round(100 * n_missing / n_cells, 2) if n_cells else None,
        "time_col": table.time_col,
        "time_freq": table.time_freq,
        "entity_col": table.entity_col,
        "value_cols": table.value_cols,
        "key": table.key,
        "sentinels_replaced": dict(sorted(table.sentinels.items())),
        "notes": table.notes,
    }
    if table.time_col:
        out["period_all_rows"] = _period_range(frame[table.time_col], table.time_freq)
        if table.value_cols:
            measured = frame[table.value_cols].notna().any(axis=1)
            out["period_with_values"] = _period_range(
                frame.loc[measured, table.time_col], table.time_freq
            )
    if table.entity_col:
        out["n_entities"] = int(frame[table.entity_col].nunique())
        out["rows_without_entity"] = int(frame[table.entity_col].isna().sum())
    out["column_profiles"] = {str(c): profile_column(frame[c]) for c in frame.columns}
    return out


# ---------------------------------------------------------------- ERA5 gridded data
class _Running:
    """Streaming min/max/count/NaN/negative and cos(latitude)-weighted monthly means."""

    def __init__(self) -> None:
        self.min = np.inf
        self.max = -np.inf
        self.n = 0
        self.n_nan = 0
        self.n_negative = 0
        self.monthly_means: list[float] = []

    def add(self, grid: np.ndarray, weights: np.ndarray) -> None:
        finite = np.isfinite(grid)
        self.n += grid.size
        self.n_nan += int((~finite).sum())
        if finite.any():
            vals = grid[finite]
            self.min = min(self.min, float(vals.min()))
            self.max = max(self.max, float(vals.max()))
            self.n_negative += int((vals < 0).sum())
            w = np.broadcast_to(weights, grid.shape)[finite]
            self.monthly_means.append(float(np.average(vals, weights=w)))

    def summary(self) -> dict[str, Any]:
        means = np.asarray(self.monthly_means)
        return {
            "n_values": self.n,
            "n_nan": self.n_nan,
            "n_negative": self.n_negative,
            "min": _num(self.min) if self.n > self.n_nan else None,
            "max": _num(self.max) if self.n > self.n_nan else None,
            "area_weighted_mean": _num(means.mean()) if means.size else None,
            "area_weighted_monthly_min": _num(means.min()) if means.size else None,
            "area_weighted_monthly_max": _num(means.max()) if means.size else None,
        }


def inspect_netcdf(path: Path, member: str, block: int = 12) -> tuple[dict[str, Any], list[Any]]:
    """Describe one ERA5 NetCDF member: dims, coordinates, units, timestamps, value ranges.

    Values are read ``block`` time steps at a time (~50 MB for 12 global 0.25-degree
    fields), so memory stays bounded whatever the file size. Returns the summary and the
    calendar months covered.
    """
    with xr.open_dataset(path) as ds:
        times = pd.DatetimeIndex(ds["valid_time"].to_numpy())
        lat = ds["latitude"].to_numpy()
        lon = ds["longitude"].to_numpy()
        out: dict[str, Any] = {
            "member": member,
            "dims": {str(k): int(v) for k, v in ds.sizes.items()},
            "coords": sorted(str(c) for c in ds.coords),
            "time_first": str(times.min().date()),
            "time_last": str(times.max().date()),
            "n_times": len(times),
            "n_distinct_months": int(times.to_period("M").nunique()),
            "time_of_day": sorted({t.strftime("%H:%M") for t in times}),
            "latitude": {"first": float(lat[0]), "last": float(lat[-1]), "n": int(lat.size)},
            "longitude": {"first": float(lon[0]), "last": float(lon[-1]), "n": int(lon.size)},
            "grid_step_deg": float(abs(lat[1] - lat[0])) if lat.size > 1 else None,
            "global_attrs": {k: str(v) for k, v in ds.attrs.items() if k in ("Conventions",)},
        }
        if "expver" in ds.coords or "expver" in ds:
            expver = pd.Series(np.asarray(ds["expver"].to_numpy()).astype(str).ravel())
            out["expver_counts"] = {str(k): int(v) for k, v in expver.value_counts().items()}
        weights = np.cos(np.deg2rad(lat))[:, None]
        variables: dict[str, Any] = {}
        for var in ds.data_vars:
            da = ds[var]
            if "valid_time" not in da.dims:
                continue
            running = _Running()
            for start in range(0, da.sizes["valid_time"], block):
                values = da.isel(valid_time=slice(start, start + block)).to_numpy()
                for grid in values:
                    running.add(np.asarray(grid, dtype="float64"), weights)
            variables[str(var)] = {
                "units": da.attrs.get("units"),
                "long_name": da.attrs.get("long_name"),
                "dims": [str(d) for d in da.dims],
                "dtype": str(da.dtype),
                **running.summary(),
            }
        out["variables"] = variables
    return out, list(times.to_period("M"))


def inspect_era5(entries: list[ManifestEntry], root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Inspect every ERA5 ZIP chunk member by member (extracted to a temporary folder)."""
    chunks = []
    months: dict[str, list[pd.Period]] = {}
    for entry in sorted(entries, key=lambda e: e.file):
        logger.info("ERA5: inspecting %s", entry.file)
        members = []
        with zipfile.ZipFile(root / entry.file) as archive, tempfile.TemporaryDirectory() as tmp:
            for info in archive.infolist():
                extracted = Path(archive.extract(info, tmp))
                member, stamps = inspect_netcdf(extracted, info.filename)
                members.append(member)
                for var in member["variables"]:
                    months.setdefault(var, []).extend(stamps)
        chunks.append({"file": entry.file, "members": members})
    coverage = {}
    for var, periods in months.items():
        index = pd.PeriodIndex(periods)
        full = pd.period_range(index.min(), index.max(), freq="M")
        coverage[var] = {
            "first": str(index.min()),
            "last": str(index.max()),
            "n_months": len(index),
            "n_duplicated": int(index.duplicated().sum()),
            "missing_months": [str(p) for p in full.difference(index)],
        }
    return {"chunks": chunks, "coverage": coverage}


# ---------------------------------------------------------------- reports
def _declared(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: meta[k] for k in DECLARED_FIELDS if k in meta}


def _fmt(x: object) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def _period_text(period: dict[str, Any] | None) -> str:
    if not period or period["first"] is None:
        return "-"
    return f"{period['first']} .. {period['last']} ({period['n_periods']})"


def render_dataset(name: str, result: dict[str, Any]) -> str:
    """Render one dataset's profile as Markdown."""
    lines = [f"# Raw profile: `{name}`", "", "Generated by `make profile` (Step 3.1).", ""]
    lines += ["## Declared metadata (`config/datasets.yaml`)", ""]
    lines += [f"- **{k}**: {v}" for k, v in result["declared"].items()] or ["- (none)"]
    lines += ["", "## Source files (manifest)", ""]
    files = result["files"]
    shown = files if len(files) <= 10 else files[:10]
    lines += [
        f"- `{f['file']}`: {f['size_bytes']:,} bytes, sha256 `{f['sha256'][:12]}...`" for f in shown
    ]
    if len(files) > len(shown):
        lines.append(f"- ... and {len(files) - len(shown)} more (see profile.json)")
    for table in result.get("tables", []):
        lines += ["", f"## Table `{table['table']}`", ""]
        lines.append(
            f"- Rows x columns: {table['rows']:,} x {table['columns']} "
            f"({table['cells']:,} cells, {table['pct_missing_cells']}% missing)"
        )
        if table["time_col"]:
            lines.append(f"- Period, all rows: {_period_text(table.get('period_all_rows'))}")
        if table["time_col"] and table["value_cols"]:
            lines.append(
                f"- Period with values in {table['value_cols']}: "
                f"{_period_text(table.get('period_with_values'))}"
            )
        if table["entity_col"]:
            lines.append(
                f"- Entities (`{table['entity_col']}`): {table['n_entities']:,} "
                f"(rows without entity: {table['rows_without_entity']:,})"
            )
        if table["key"]:
            lines.append(f"- Expected key (tested in Step 3.2): `{table['key']}`")
        sentinels = ", ".join(f"`{k}` x{v:,}" for k, v in table["sentinels_replaced"].items())
        lines.append(f"- Missing codes replaced at read: {sentinels or 'none'}")
        lines += [f"- Note: {n}" for n in table["notes"]]
        lines += [
            "",
            "| Column | Type | Missing % | Unique | Min | Median | Max | Top values |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for col, p in table["column_profiles"].items():
            top = ", ".join(f"{k} ({v:,})" for k, v in p.get("top_values", {}).items())
            if p["kind"] == "text" and "top_values" not in p:
                top = f"not listed (> {MAX_LISTED_CATEGORIES} distinct)"
            lines.append(
                f"| `{col}` | {p['kind']} ({p['dtype']}) | {_fmt(p['pct_missing'])} "
                f"| {p['n_unique']:,} | {_fmt(p.get('min'))} | {_fmt(p.get('median'))} "
                f"| {_fmt(p.get('max'))} | {top.replace('|', '/')} |"
            )
    if "era5" in result:
        lines += render_era5(result["era5"])
    return "\n".join(lines) + "\n"


def render_era5(era5: dict[str, Any]) -> list[str]:
    """Render the ERA5 inspection: coverage per variable, then one row per chunk member."""
    lines = ["", "## Coverage per variable", "", "| Variable | First | Last | Months | "]
    lines[-1] += "Duplicated | Missing months |"
    lines.append("|---|---|---|---|---|---|")
    for var, c in era5["coverage"].items():
        missing = ", ".join(c["missing_months"]) or "none"
        lines.append(
            f"| `{var}` | {c['first']} | {c['last']} | {c['n_months']} "
            f"| {c['n_duplicated']} | {missing} |"
        )
    lines += [
        "",
        "## Members",
        "",
        "| Member | Dims | Time of day | Lat first/last | Lon first/last | expver | Variable "
        "| Units | NaN | Negative | Min | Max | Area-weighted mean |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for chunk in era5["chunks"]:
        for m in chunk["members"]:
            for var, v in m["variables"].items():
                lines.append(
                    f"| `{Path(m['member']).name}` | {m['dims']} | {m['time_of_day']} "
                    f"| {m['latitude']['first']}/{m['latitude']['last']} "
                    f"| {m['longitude']['first']}/{m['longitude']['last']} "
                    f"| {m.get('expver_counts', '-')} | `{var}` | {v['units']} | {v['n_nan']:,} "
                    f"| {v['n_negative']:,} | {_fmt(v['min'])} | {_fmt(v['max'])} "
                    f"| {_fmt(v['area_weighted_mean'])} |"
                )
    return lines


def render_inventory(results: dict[str, dict[str, Any]]) -> str:
    """One line per table across all datasets (input to the Step 3.6 report)."""
    lines = [
        "# Raw data inventory (Step 3.1)",
        "",
        "| Dataset | Table | Rows | Cols | Cells | Missing % | Period with values | Entities "
        "| Missing codes replaced |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, result in results.items():
        for t in result.get("tables", []):
            replaced = sum(t["sentinels_replaced"].values())
            lines.append(
                f"| `{name}` | {t['table']} | {t['rows']:,} | {t['columns']} | {t['cells']:,} "
                f"| {t['pct_missing_cells']} | {_period_text(t.get('period_with_values'))} "
                f"| {t.get('n_entities', '-')} | {replaced:,} |"
            )
        if "era5" in result:
            for var, c in result["era5"]["coverage"].items():
                lines.append(
                    f"| `{name}` | {var} (grid) | - | - | - | - "
                    f"| {c['first']} .. {c['last']} ({c['n_months']}) | grid | - |"
                )
    return "\n".join(lines) + "\n"


def _jsonable(x: object) -> object:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.integer | np.floating):
        return _num(x)
    if isinstance(x, pd.Timestamp):
        return _stamp(x)
    return x


def profile_dataset(
    name: str, manifest: list[ManifestEntry], catalogue: dict[str, Any], root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Profile one dataset (tables, or the ERA5 grid) with its declared metadata and files."""
    entries = sorted((e for e in manifest if e.dataset == name), key=lambda e: e.file)
    result: dict[str, Any] = {
        "declared": _declared(catalogue.get(name, {})),
        "files": [
            {"file": e.file, "size_bytes": e.size_bytes, "sha256": e.sha256} for e in entries
        ],
    }
    if name == ERA5:
        result["era5"] = inspect_era5(entries, root)
    else:
        result["tables"] = [profile_table(t) for t in read_dataset(name, manifest, root)]
    return result


def main(argv: list[str] | None = None) -> int:
    """Profile the requested datasets (default: all) and write the reports."""
    parser = argparse.ArgumentParser(description="Step 3.1 raw-data profiling.")
    parser.add_argument("datasets", nargs="*", help="dataset names (default: all)")
    parser.add_argument("--skip-era5", action="store_true", help="skip the slow ERA5 grid scan")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "profiling")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    names = args.datasets or [*READERS, ERA5]
    if args.skip_era5:
        names = [n for n in names if n != ERA5]
    unknown = sorted(set(names) - {*READERS, ERA5})
    if unknown:
        parser.error(f"unknown dataset(s): {unknown}")

    manifest = load_manifest()
    catalogue = load_datasets()
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "profile.json"
    previous = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    results: dict[str, dict[str, Any]] = dict(previous.get("datasets", {}))
    for name in names:
        result = _jsonable(profile_dataset(name, manifest, catalogue))
        if not isinstance(result, dict):
            raise TypeError(f"{name}: profile is not a mapping")
        results[name] = result
        (args.out / f"{name}.md").write_text(
            render_dataset(name, result), encoding="utf-8", newline="\n"
        )
        logger.info("%s: profiled", name)

    ordered = {n: results[n] for n in [*READERS, ERA5] if n in results}
    payload = {"manifest_sha256": sha256_file(get_path("manifest")), "datasets": ordered}
    json_path.write_text(
        json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8", newline="\n"
    )
    (args.out / "inventory.md").write_text(
        render_inventory(ordered), encoding="utf-8", newline="\n"
    )
    logger.info("wrote %s (%d dataset(s))", args.out, len(ordered))
    return 0


if __name__ == "__main__":
    sys.exit(main())
