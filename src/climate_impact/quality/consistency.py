"""Step 3.5: cross-source consistency (DEC-078).

Global records (GISTEMP, HadCRUT5, Berkeley)
    Each series is rebased per calendar month to 1991-2020 (DEC-009, offsets checked in
    Step 3.3). Agreement is measured on the common window with correlations of levels and of
    *first differences* (year-to-year changes; a shared trend cannot inflate them) and with
    pairwise differences per period, expected to be largest in the sparsely observed decades.
ERA5 vs Berkeley by country
    ERA5 2 m temperature is averaged over each country of a fixed sample with cos(latitude)
    weights on Natural Earth 1:50m masks (cell centres). Annual anomalies of both sources,
    each rebased to 1991-2020, are compared on 1961-2020. Risk RD5 (a step in the difference
    at the start of satellite assimilation, 1979) is tested with OLS of the difference on a
    post-1979 indicator, Newey-West standard errors and Holm correction across countries.
    A step test cannot tell a discontinuity from a gradual drift; either triggers the
    pre-registered RD5 response (robustness on 1979-2025).
Blinding
    No warming trend is estimated; only differences and agreement between sources.

Natural Earth ISO codes: ``ISO_A3`` is -99 for France, Norway and Kosovo; see ``iso3_codes``

Usage: ``python -m climate_impact.quality.consistency`` (``make consistency``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import zipfile
from itertools import combinations
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from matplotlib.figure import Figure
from statsmodels.stats.multitest import multipletests

from climate_impact.config import PROJECT_ROOT, load_config
from climate_impact.ingest.readers import RawTable, read_dataset
from climate_impact.provenance import ManifestEntry, load_manifest
from climate_impact.quality.checks import GOOD, PROBLEMATIC, load_rules

logger = logging.getLogger(__name__)

ERA5 = "era5_monthly_single_levels"
ACCEPTABLE = "Acceptable"
NE_VERSION = "v5.1.2"
ISO_FIXES = {"KOS": "XKX"}
# Validated categorical slots 1-3 (all-pairs CVD-safe); text stays in ink colours.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e6e5e1"


def grade_high(value: float, thresholds: dict[str, float]) -> str:
    """Grade a metric where larger is better (correlations)."""
    if value >= thresholds["good"]:
        return GOOD
    if value >= thresholds["acceptable"]:
        return ACCEPTABLE
    return PROBLEMATIC


# ---------------------------------------------------------------- series helpers
def rebase_monthly(frame: pd.DataFrame, value: str, period: tuple[int, int]) -> pd.Series:
    """Subtract each calendar month's mean over ``period`` (needs every year of it)."""
    month = frame["date"].dt.month
    inside = frame["date"].dt.year.between(*period)
    counts = frame.loc[inside, value].notna().groupby(month[inside]).sum()
    expected = period[1] - period[0] + 1
    if len(counts) != 12 or (counts < expected).any():
        raise ValueError(f"rebasing needs {expected} values per calendar month in {period}")
    climatology = frame.loc[inside, value].groupby(month[inside]).mean()
    return pd.Series(frame[value] - month.map(climatology), index=frame.index)


def annual_means(dates: pd.Series, values: pd.Series, min_months: int = 12) -> pd.Series:
    """Calendar-year means; NaN unless ``min_months`` months are valid (plan section 3.2)."""
    frame = pd.DataFrame({"year": dates.dt.year, "v": values})
    grouped = frame.groupby("year")["v"]
    means = grouped.mean().where(grouped.count() >= min_months)
    return means.dropna()


def _corr(a: pd.Series, b: pd.Series) -> float | None:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3:
        return None
    return round(float(pair.iloc[:, 0].corr(pair.iloc[:, 1])), 4)


def newey_west_lags(n: int) -> int:
    """Newey-West rule of thumb, floor(4 (T/100)^(2/9)) (config: hac_maxlags)."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


# ---------------------------------------------------------------- global records
def global_comparison(
    series: dict[str, pd.DataFrame],
    window: tuple[int, int],
    periods: list[list[int]],
    rebase: tuple[int, int],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Correlations and pairwise differences of rebased global anomaly series."""
    monthly = {}
    for name, frame in series.items():
        rebased = rebase_monthly(frame, "anomaly", rebase)
        monthly[name] = pd.Series(rebased.to_numpy(), index=frame["date"].to_numpy())
    wide = pd.DataFrame(monthly)
    years = pd.DatetimeIndex(wide.index).year
    wide = wide[(years >= window[0]) & (years <= window[1])]
    annual = pd.DataFrame(
        {n: annual_means(pd.Series(wide.index), pd.Series(wide[n].to_numpy())) for n in wide}
    )
    out: dict[str, Any] = {"window": list(window), "pairs": {}}
    for a, b in combinations(wide.columns, 2):
        diff = (annual[a] - annual[b]).dropna()
        by_period = {}
        for lo, hi in periods:
            part = diff[(diff.index >= lo) & (diff.index <= hi)]
            by_period[f"{lo}-{hi}"] = {
                "mean": round(float(part.mean()), 4),
                "sd": round(float(part.std()), 4),
                "rms": round(float(np.sqrt((part**2).mean())), 4),
            }
        diff_corr = _corr(annual[a].diff(), annual[b].diff())
        out["pairs"][f"{a}-{b}"] = {
            "monthly_corr_levels": _corr(wide[a], wide[b]),
            "monthly_corr_changes": _corr(wide[a].diff(), wide[b].diff()),
            "annual_corr_levels": _corr(annual[a], annual[b]),
            "annual_corr_changes": diff_corr,
            "annual_difference_by_period": by_period,
            "largest_abs_difference": {
                "year": int(diff.index.to_numpy()[int(np.argmax(np.abs(diff.to_numpy())))]),
                "degc": round(float(diff.to_numpy()[int(np.argmax(np.abs(diff.to_numpy())))]), 4),
            },
            "score": grade_high(diff_corr or 0.0, thresholds),
        }
    out["annual"] = annual
    return out


# ---------------------------------------------------------------- country masks
def natural_earth_countries() -> gpd.GeoDataFrame:
    """Natural Earth 1:50m admin-0 countries with a clean ``iso3`` column (DEC-060 source)."""
    from regionmask.defined_regions import _natural_earth  # pinned regionmask 0.13.0

    frame = _natural_earth._countries_50.read(NE_VERSION)
    frame["iso3"] = iso3_codes(frame)
    return frame


def iso3_codes(frame: pd.DataFrame) -> pd.Series:
    """One unique ISO3-like code per polygon.

    ``ISO_A3`` first; where it is -99 (France, Norway, Kosovo, some territories) use
    ``ISO_A3_EH`` unless that code already belongs to another polygon (``ISO_A3_EH`` maps
    Christmas/Cocos and Ashmore-Cartier to AUS), else ``ADM0_A3``; ``KOS`` -> ``XKX``.
    """
    primary = frame["ISO_A3"].astype("string")
    taken = set(primary[primary != "-99"])
    extended = frame["ISO_A3_EH"].astype("string")
    use_eh = (primary == "-99") & (extended != "-99") & ~extended.isin(taken)
    codes = primary.where(primary != "-99", extended.where(use_eh, frame["ADM0_A3"]))
    codes = codes.replace(ISO_FIXES)
    duplicated = codes[codes.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Natural Earth codes not unique: {duplicated}")
    return codes


def country_cells(
    countries: gpd.GeoDataFrame, iso3s: list[str], lat: np.ndarray, lon: np.ndarray
) -> dict[str, dict[str, Any]]:
    """Flat grid indices, cos(lat) weights, cell count and polygon area per country."""
    import regionmask

    subset = countries[countries["iso3"].isin(iso3s)].reset_index(drop=True)
    missing = sorted(set(iso3s) - set(subset["iso3"]))
    if missing:
        raise ValueError(f"countries not in Natural Earth: {missing}")
    regions = regionmask.from_geopandas(subset, names="NAME", abbrevs="iso3")
    mask = regions.mask_3D(lon, lat)
    weights = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size))
    area = subset.to_crs("EPSG:6933").set_index("iso3").geometry.area / 1e6
    out = {}
    for code, layer in zip(mask["abbrevs"].to_numpy(), mask.transpose("region", ...), strict=True):
        flat = np.flatnonzero(layer.to_numpy())
        out[str(code)] = {
            "index": flat,
            "weights": weights.ravel()[flat],
            "n_cells": int(flat.size),
            "ne_area_km2": round(float(area.loc[code]), 0),
        }
    return out


def era5_country_monthly(
    entries: list[ManifestEntry],
    root: Path,
    iso3s: list[str],
    countries: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Area-weighted monthly t2m (degC) per country from the ERA5 ZIP chunks."""
    rows: list[dict[str, Any]] = []
    cells: dict[str, dict[str, Any]] = {}
    for entry in sorted(entries, key=lambda e: e.file):
        logger.info("ERA5 country means: %s", entry.file)
        with zipfile.ZipFile(root / entry.file) as archive, tempfile.TemporaryDirectory() as tmp:
            for info in archive.infolist():
                with xr.open_dataset(Path(archive.extract(info, tmp))) as ds:
                    if "t2m" not in ds:
                        continue
                    if not cells:
                        lat, lon = ds["latitude"].to_numpy(), ds["longitude"].to_numpy()
                        cells = country_cells(countries, iso3s, lat, lon)
                    stamps = pd.DatetimeIndex(ds["valid_time"].to_numpy()).to_period("M")
                    for k in range(0, len(stamps), 12):
                        block = ds["t2m"].isel(valid_time=slice(k, k + 12)).to_numpy()
                        for t, grid in enumerate(block):
                            flat = grid.ravel()
                            for code, c in cells.items():
                                value = np.average(flat[c["index"]], weights=c["weights"])
                                rows.append(
                                    {
                                        "iso3": code,
                                        "date": stamps[k + t].to_timestamp(),
                                        "t2m_c": float(value) - 273.15,
                                    }
                                )
    return pd.DataFrame(rows), cells


# ---------------------------------------------------------------- ERA5 vs Berkeley
def annual_anomalies(frame: pd.DataFrame, value: str, rebase: tuple[int, int]) -> pd.DataFrame:
    """Per country: annual means (12 valid months) minus their 1991-2020 mean."""
    parts = []
    for code, part in frame.groupby("iso3"):
        annual = annual_means(part["date"], part[value])
        base = annual[(annual.index >= rebase[0]) & (annual.index <= rebase[1])]
        if len(base) < rebase[1] - rebase[0] + 1:
            continue
        parts.append(
            pd.DataFrame({"iso3": code, "year": annual.index, "anom": annual - base.mean()})
        )
    empty = pd.DataFrame(
        {
            "iso3": pd.Series(dtype="string"),
            "year": pd.Series(dtype="int64"),
            "anom": pd.Series(dtype="float64"),
        }
    )
    return pd.concat(parts, ignore_index=True) if parts else empty


def step_test(diff: pd.Series, split: int) -> dict[str, float]:
    """OLS of the difference on a post-split indicator, Newey-West standard errors."""
    post = (diff.index >= split).astype(float)
    exog = sm.add_constant(pd.Series(post, index=diff.index, name="post"))
    lags = newey_west_lags(len(diff))
    fit = sm.OLS(diff.to_numpy(), exog.to_numpy()).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {
        "step_degc": round(float(fit.params[1]), 4),
        "se": round(float(fit.bse[1]), 4),
        "p": round(float(fit.pvalues[1]), 4),
        "hac_lags": lags,
    }


def country_comparison(
    era5: pd.DataFrame,
    berkeley: pd.DataFrame,
    window: tuple[int, int],
    split: int,
    tolerance: float,
    alpha: float,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Agreement and RD5 step test per country; Holm across countries."""
    merged = era5.merge(berkeley, on=["iso3", "year"], suffixes=("_era5", "_bk"))
    merged = merged[merged["year"].between(*window)]
    results: dict[str, Any] = {}
    for code, part in merged.groupby("iso3"):
        part = part.set_index("year").sort_index()
        diff = part["anom_era5"] - part["anom_bk"]
        corr_changes = _corr(part["anom_era5"].diff(), part["anom_bk"].diff())
        results[str(code)] = {
            "n_years": len(part),
            "corr_levels": _corr(part["anom_era5"], part["anom_bk"]),
            "corr_changes": corr_changes,
            "rmsd_degc": round(float(np.sqrt((diff**2).mean())), 4),
            "rd5": step_test(diff, split),
            "score": grade_high(corr_changes or 0.0, thresholds),
        }
    if results:
        pvals = [r["rd5"]["p"] for r in results.values()]
        holm = multipletests(pvals, alpha=alpha, method="holm")[1]
        for r, p in zip(results.values(), holm, strict=True):
            r["rd5"]["p_holm"] = round(float(p), 4)
            r["rd5"]["tripwire"] = bool(abs(r["rd5"]["step_degc"]) >= tolerance and p < alpha)
    return results


# ---------------------------------------------------------------- figures
def global_difference_figure(annual: pd.DataFrame, path: Path) -> None:
    """Pairwise annual differences between the rebased global records, direct-labelled."""
    fig = Figure(figsize=(9, 3.6), dpi=150)
    ax = fig.add_subplot()
    ax.axhline(0, color=MUTED, linewidth=0.8)
    for color, (a, b) in zip(SERIES_COLORS, combinations(annual.columns, 2), strict=False):
        diff = (annual[a] - annual[b]).dropna()
        ax.plot(diff.index, diff.to_numpy(), color=color, linewidth=1.6)
        ax.annotate(
            f"{a} - {b}",
            (diff.index[-1], diff.iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=7,
            color=INK,
            va="center",
        )
    ax.set_ylabel("difference (degC)", fontsize=7, color=INK)
    ax.tick_params(labelsize=7, colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(
        "Global records: annual differences after rebasing to 1991-2020 (Step 3.5)",
        fontsize=9,
        color=INK,
        loc="left",
    )
    fig.subplots_adjust(left=0.08, right=0.84, top=0.88, bottom=0.12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Software": None})


def rd5_figure(results: dict[str, Any], tolerance: float, split: int, path: Path) -> None:
    """ERA5 - Berkeley step at the split year per country, 95 % CI, tolerance band."""
    order = sorted(results, key=lambda c: results[c]["rd5"]["step_degc"])
    steps = np.array([results[c]["rd5"]["step_degc"] for c in order])
    half = 1.96 * np.array([results[c]["rd5"]["se"] for c in order])
    fig = Figure(figsize=(6.5, 0.28 * len(order) + 1.2), dpi=150)
    ax = fig.add_subplot()
    ax.axvspan(-tolerance, tolerance, color=GRID, zorder=0)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    y = np.arange(len(order))
    ax.errorbar(
        steps,
        y,
        xerr=half,
        fmt="o",
        color=SERIES_COLORS[0],
        markersize=4,
        elinewidth=1.2,
        capsize=0,
    )
    ax.set_yticks(y, order, fontsize=7)
    ax.tick_params(labelsize=7, colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(
        f"step in ERA5 - Berkeley from {split} (degC, 95% CI); band = +-{tolerance} tolerance",
        fontsize=7,
        color=INK,
    )
    ax.set_title(
        "Risk RD5: satellite-era step by country (Step 3.5)", fontsize=9, color=INK, loc="left"
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        top=1 - 0.45 / (0.28 * len(order) + 1.2),
        bottom=0.55 / (0.28 * len(order) + 1.2),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Software": None})


# ---------------------------------------------------------------- orchestration
def run(
    manifest: list[ManifestEntry], root: Path, cfg: dict[str, Any], figures: Path
) -> dict[str, Any]:
    """Run both comparisons, draw the figures, return the evidence."""
    spec = load_rules()["consistency"]
    rebase = (
        int(cfg["analysis"]["baseline_period"][0]),
        int(cfg["analysis"]["baseline_period"][1]),
    )
    alpha = float(cfg["statistics"]["alpha"])
    cache: dict[str, dict[str, RawTable]] = {}

    def table(key: str) -> RawTable:
        dataset, tab = key.split("/")
        if dataset not in cache:
            cache[dataset] = {t.table: t for t in read_dataset(dataset, manifest, root)}
        return cache[dataset][tab]

    series = {name: table(key).frame for name, key in spec["global_series"].items()}
    window = (int(spec["global_window"][0]), int(spec["global_window"][1]))
    glob = global_comparison(series, window, spec["periods"], rebase, spec["global_diff_corr"])
    annual = glob.pop("annual")
    global_difference_figure(annual, figures / "consistency_global_differences.png")

    iso3s = [str(c) for c in spec["countries"]]
    countries = natural_earth_countries()
    era5_monthly, cells = era5_country_monthly(
        [e for e in manifest if e.dataset == ERA5], root, iso3s, countries
    )
    bk = table("berkeley_country_monthly/monthly").frame
    bk = bk[bk["iso3"].isin(iso3s)]
    headers = table("berkeley_country_monthly/file_headers").frame.set_index("iso3")
    era5_anom = annual_anomalies(era5_monthly, "t2m_c", rebase)
    bk_anom = annual_anomalies(bk, "anomaly", rebase)
    cwin = (int(spec["country_window"][0]), int(spec["country_window"][1]))
    split, tol = int(spec["rd5_split_year"]), float(spec["rd5_step_tolerance_degc"])
    per_country = country_comparison(
        era5_anom, bk_anom, cwin, split, tol, alpha, spec["country_diff_corr"]
    )
    min_cells = int(cfg["spatial"]["min_grid_cells_flag"])
    for code, res in per_country.items():
        c = cells[code]
        res["era5_cells"] = c["n_cells"]
        res["few_cells_flag"] = c["n_cells"] < min_cells
        res["ne_area_km2"] = c["ne_area_km2"]
        areas = headers["area_km2"].to_dict()
        bk_area = float(areas[code]) if code in areas else None
        res["berkeley_area_km2"] = bk_area
        res["area_ratio_berkeley_to_ne"] = round(bk_area / c["ne_area_km2"], 3) if bk_area else None
    rd5_figure(per_country, tol, split, figures / "consistency_rd5_steps.png")
    return {"global": glob, "countries": per_country}


def render(result: dict[str, Any]) -> str:
    """Markdown summary tables plus the full evidence."""
    lines = [
        "# Cross-source consistency (Step 3.5)",
        "",
        "Generated by `make consistency` (rules: `config/quality_rules.yaml`, DEC-078).",
        "Sources are compared with each other only; no warming trend is estimated",
        "(H1 is tested in Phase 6).",
        "",
        "## Global records (rebased to 1991-2020)",
        "",
        "![global differences](../figures/consistency_global_differences.png)",
        "",
        "| Pair | Annual corr (levels) | Annual corr (changes) | RMS diff 1880-1919 "
        "| RMS diff 1980-2024 | Largest diff | Score |",
        "|---|---|---|---|---|---|---|",
    ]
    for pair, r in result["global"]["pairs"].items():
        periods = r["annual_difference_by_period"]
        first, last = next(iter(periods.values())), list(periods.values())[-1]
        big = r["largest_abs_difference"]
        lines.append(
            f"| {pair} | {r['annual_corr_levels']} | {r['annual_corr_changes']} | {first['rms']} "
            f"| {last['rms']} | {big['degc']} ({big['year']}) | {r['score']} |"
        )
    lines += [
        "",
        "## ERA5 vs Berkeley Earth by country (annual anomalies, 1961-2020)",
        "",
        "![RD5 steps](../figures/consistency_rd5_steps.png)",
        "",
        "| Country | Corr (levels) | Corr (changes) | RMSD | RD5 step | Holm p | Tripwire "
        "| ERA5 cells | Area ratio (Berkeley/NE) | Score |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for code, r in sorted(result["countries"].items()):
        s = r["rd5"]
        lines.append(
            f"| {code} | {r['corr_levels']} | {r['corr_changes']} | {r['rmsd_degc']} "
            f"| {s['step_degc']} | {s['p_holm']} | {'YES' if s['tripwire'] else 'no'} "
            f"| {r['era5_cells']}{' (few)' if r['few_cells_flag'] else ''} "
            f"| {r['area_ratio_berkeley_to_ne']} | {r['score']} |"
        )
    lines += ["", "## Full evidence", "", "```json", json.dumps(result, indent=1), "```", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run Step 3.5 and write ``reports/quality/consistency.{json,md}`` plus figures."""
    parser = argparse.ArgumentParser(description="Step 3.5 cross-source consistency.")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "quality")
    parser.add_argument("--figures", type=Path, default=PROJECT_ROOT / "reports" / "figures")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(load_manifest(), PROJECT_ROOT, load_config(), args.figures)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "consistency.json").write_text(
        json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    (args.out / "consistency.md").write_text(render(result), encoding="utf-8", newline="\n")
    logger.info("wrote %s", args.out / "consistency.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
