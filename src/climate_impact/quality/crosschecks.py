"""Step 3.2b: consistency and integrity checks, ERA5 extremes, EM-DAT reporting lag.

Consistency (within a dataset, and between the data and ``config.yaml``)
    * FAOSTAT: one unit per element; reported yield = production / area within a tolerance;
      blank values carry flag ``M``.
    * NOAA CO2: the three ancillary missing codes occur in the same months.
    * Code <-> name is one-to-one (OWID, WDI, EM-DAT).
    * Every category or code named in ``config.yaml`` exists in the data (a misspelled
      subgroup would silently select nothing).
Integrity (between datasets)
    Each dataset's country codes are classified against the reference universe (the
    economies with a ``Region`` in ``WDICountry.csv``): in reference / valid ISO3 outside it
    (territories) / historical / invalid. Only *invalid* codes are an integrity error;
    coverage of the reference is reported for the Step 3.6 sample definition. FAOSTAT area
    codes are mapped to ISO3 by M49 first and FAO code second (``country_converter``'s
    bundled table; its API is not used because it fails under pandas 3, DEC-065).
ERA5 extremes
    Grid cells holding the 5 highest (and, for t2m, lowest) monthly values, with dates.
EM-DAT reporting lag
    Share of events of mature years entered later than the last year's events could have
    been at export time; estimates how incomplete the final year still is.

Usage: ``python -m climate_impact.quality.crosschecks [--skip-era5]`` (``make crosschecks``).
"""

from __future__ import annotations

import argparse
import importlib.resources
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

from climate_impact.config import PROJECT_ROOT, load_config
from climate_impact.ingest.readers import read_dataset
from climate_impact.provenance import ManifestEntry, load_manifest
from climate_impact.quality.checks import (
    ACCEPTABLE,
    GOOD,
    PROBLEMATIC,
    _token_values,
    aggregate_mask,
    grade,
    load_rules,
    select_rows,
)

logger = logging.getLogger(__name__)
N_EXAMPLES = 5
ERA5 = "era5_monthly_single_levels"
WORST = {GOOD: 0, ACCEPTABLE: 1, PROBLEMATIC: 2}


def worst(scores: list[str]) -> str:
    """Return the worst of several scores (Problematic > Acceptable > Good)."""
    return max(scores, key=WORST.__getitem__) if scores else GOOD


# ---------------------------------------------------------------- country reference
def country_table() -> pd.DataFrame:
    """``country_converter``'s bundled table, read as text (blank = missing, 'NA' = Namibia)."""
    source = importlib.resources.files("country_converter") / "country_data.tsv"
    with importlib.resources.as_file(source) as path:
        table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for col in ("UNcode", "FAOcode"):
        table[col] = table[col].str.lstrip("0")
    return table


def _norm_code(values: pd.Series) -> pd.Series:
    """FAOSTAT codes as plain digits: ``'004`` -> ``4`` (apostrophe and zero padding off)."""
    return values.astype("string").str.strip().str.lstrip("'").str.lstrip("0")


def fao_to_iso3(areas: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Map FAO areas to ISO3: M49 first, FAO code second; record source and disagreement."""
    by_m49 = dict(zip(table["UNcode"], table["ISO3"], strict=True))
    by_fao = dict(zip(table["FAOcode"], table["ISO3"], strict=True))
    by_m49.pop("", None)
    by_fao.pop("", None)
    out = areas.copy()
    m49 = _norm_code(out["Area Code (M49)"]).map(by_m49)
    fao = _norm_code(out["Area Code"]).map(by_fao)
    out["iso3"] = m49.fillna(fao)
    out["iso3_source"] = np.where(m49.notna(), "m49", np.where(fao.notna(), "fao_code", "none"))
    out["sources_disagree"] = (m49.notna() & fao.notna() & (m49 != fao)).to_numpy(dtype=bool)
    return out


def classify_codes(
    codes: pd.Series,
    names: pd.Series,
    reference: set[str],
    table: pd.DataFrame,
    historical: dict[str, str],
    scoring: dict[str, Any],
) -> dict[str, Any]:
    """Classify a dataset's country codes and score its integrity (invalid-code share)."""
    pairs = pd.DataFrame({"code": codes.astype("string"), "name": names.astype("string")})
    pairs = pairs.dropna(subset=["code"]).drop_duplicates("code").sort_values("code")
    valid_iso = set(table["ISO3"]) - {""}
    obsolete = set(table.loc[table["obsolete"] != "", "ISO3"])
    groups: dict[str, list[str]] = {
        "in_reference": [],
        "territory_or_other": [],
        "historical": [],
        "invalid": [],
    }
    for code, name in zip(pairs["code"], pairs["name"], strict=True):
        label = f"{code} ({name})"
        if code in reference:
            groups["in_reference"].append(code)
        elif code in historical or code in obsolete:
            groups["historical"].append(label)
        elif code in valid_iso:
            groups["territory_or_other"].append(label)
        else:
            groups["invalid"].append(label)
    n = len(pairs)
    pct_invalid = round(100 * len(groups["invalid"]) / n, 2) if n else 0.0
    missing = sorted(reference - set(pairs["code"]))
    return {
        "n_codes": n,
        "n_in_reference": len(groups["in_reference"]),
        "reference_coverage_pct": round(100 * len(groups["in_reference"]) / len(reference), 1),
        "territory_or_other": groups["territory_or_other"],
        "historical": groups["historical"],
        "invalid": groups["invalid"],
        "reference_codes_absent": missing,
        "metric": pct_invalid,
        "score": grade(pct_invalid, scoring["integrity_invalid_pct"]),
    }


# ---------------------------------------------------------------- consistency
def faostat_consistency(frame: pd.DataFrame, cross: dict[str, Any]) -> dict[str, Any]:
    """Units per element, yield identity, and blank values versus flag ``M``."""
    elements = frame["Element Code"].astype("string")
    units = (
        frame.assign(_e=elements, _u=frame["Unit"].astype("string"))
        .groupby("_e")["_u"]
        .unique()
        .apply(sorted)
    )
    units_ok = all(len(u) == 1 for u in units) and list(units.get("5412", ["kg/ha"])) == ["kg/ha"]
    key = ["Area Code", "Item Code", "Year"]
    wide = (
        frame.assign(_e=elements)
        .pivot_table(index=key, columns="_e", values="Value", aggfunc="first", observed=True)
        .dropna(
            subset=[
                c
                for c in ("5312", "5510", "5412")
                if c in frame["Element Code"].astype(str).unique()
            ]
        )
    )
    tol = float(cross["yield_identity_tolerance"])
    result: dict[str, Any] = {"units_per_element": {k: list(v) for k, v in units.items()}}
    if {"5312", "5510", "5412"} <= set(wide.columns):
        usable = wide[(wide["5312"] > 0) & (wide["5412"] > 0)]
        computed = usable["5510"] * 1000 / usable["5312"]
        rel = (computed - usable["5412"]).abs() / usable["5412"]
        off = rel[rel > tol].sort_values(ascending=False)
        examples = [
            {
                **{k: str(v) for k, v in zip(key, idx, strict=True)},
                "reported_kg_ha": round(float(usable.loc[idx, "5412"]), 1),
                "computed_kg_ha": round(float(computed.loc[idx]), 1),
                "area_ha": float(usable.loc[idx, "5312"]),
            }
            for idx in off.index[:N_EXAMPLES]
        ]
        pct = round(100 * len(off) / max(len(rel), 1), 3)
        result["yield_identity"] = {
            "n_triples": len(rel),
            "n_off": len(off),
            "pct_off": pct,
            "median_rel_diff": round(float(rel.median()), 6),
            "examples": examples,
            "score": grade(pct, cross["yield_identity_pct"]),
        }
    high = frame[(elements == "5412") & (frame["Value"] > 30000)]
    if len(high):
        idx = high.set_index(key).index.intersection(wide.index)
        result["high_yields"] = [
            {
                **{k: str(v) for k, v in zip(key, i, strict=True)},
                "yield_kg_ha": float(wide.loc[i, "5412"]),
                "area_ha": float(wide.loc[i, "5312"]),
            }
            for i in idx[:10]
        ]
    blank = frame["Value"].isna()
    flags = frame["Flag"].astype("string")
    blank_not_m = int((blank & flags.ne("M").fillna(True)).sum())
    m_with_value = int((~blank & flags.eq("M").fillna(False)).sum())
    pct_blank = round(100 * blank_not_m / max(int(blank.sum()), 1), 3)
    result["blank_vs_flag"] = {
        "n_blank": int(blank.sum()),
        "blank_without_flag_M": blank_not_m,
        "flag_M_with_value": m_with_value,
        "flag_shares_pct": {
            str(k): round(100 * float(v), 2)
            for k, v in flags.value_counts(normalize=True, dropna=False).items()
        },
        "score": grade(pct_blank, cross["blank_flag_pct"]),
    }
    scores = [result["blank_vs_flag"]["score"], GOOD if units_ok else PROBLEMATIC]
    if "yield_identity" in result:
        scores.append(result["yield_identity"]["score"])
    result["units_ok"] = units_ok
    result["score"] = worst(scores)
    return result


def noaa_ancillary(frame: pd.DataFrame) -> dict[str, Any]:
    """Months in which the ancillary missing codes do not coincide (ndays, sdev, unc)."""
    miss = frame[["ndays", "sdev", "unc"]].isna()
    pattern = miss.astype(int).astype(str).agg("".join, axis=1)
    disagree = frame.loc[miss.any(axis=1) & ~miss.all(axis=1)]
    return {
        "missing_patterns_ndays_sdev_unc": {
            str(k): int(v) for k, v in pattern.value_counts().items()
        },
        "months_disagreeing": [
            {
                "date": str(r["date"].date()),
                **{c: None if pd.isna(r[c]) else float(r[c]) for c in ("ndays", "sdev", "unc")},
            }
            for _, r in disagree.head(10).iterrows()
        ],
        "co2_missing_in_those_months": int(
            disagree[["average", "deseasonalized"]].isna().sum().sum()
        ),
        "score": GOOD if disagree.empty else ACCEPTABLE,
    }


def one_to_one(frame: pd.DataFrame, code: str, name: str) -> dict[str, Any]:
    """Codes with several names, and names with several codes."""
    pairs = frame[[code, name]].dropna().astype("string").drop_duplicates()
    codes_many = pairs.groupby(code)[name].nunique()
    names_many = pairs.groupby(name)[code].nunique()
    multi_codes = codes_many[codes_many > 1].index.tolist()
    multi_names = names_many[names_many > 1].index.tolist()
    return {
        "codes_with_several_names": {
            c: sorted(pairs.loc[pairs[code] == c, name].tolist()) for c in multi_codes[:10]
        },
        "names_with_several_codes": {
            n: sorted(pairs.loc[pairs[name] == n, code].tolist()) for n in multi_names[:10]
        },
        "score": PROBLEMATIC if multi_names else (ACCEPTABLE if multi_codes else GOOD),
    }


def required_values(
    frame: pd.DataFrame, spec: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    """Every configured category/code must occur in the data."""
    absent = {}
    for col, values in spec.items():
        present = set(frame[col].astype("string").dropna())
        wanted = _token_values(values, cfg) if isinstance(values, str) else [str(v) for v in values]
        missing = [v for v in wanted if v not in present]
        if missing:
            absent[col] = missing
    return {"absent": absent, "score": PROBLEMATIC if absent else GOOD}


def emdat_lag(frame: pd.DataFrame, access_date: str, cross: dict[str, Any]) -> dict[str, Any]:
    """Estimate how complete the last full year is, from the entry-date history."""
    years = pd.to_numeric(frame["Start Year"], errors="coerce")
    last = pd.Timestamp(access_date).year - 1
    counts = years.value_counts().sort_index()
    prior = counts.loc[last - 9 : last - 1]
    mean, sd = float(prior.mean()), float(prior.std())
    out: dict[str, Any] = {
        "last_full_year": last,
        "events_last_year": int(counts.get(last, 0)),
        "prior_9y_mean": round(mean, 1),
        "prior_9y_sd": round(sd, 1),
        "z_last_year": round((counts.get(last, 0) - mean) / sd, 2) if sd > 0 else None,
    }
    if "Entry Date" not in frame:
        out["entry_date"] = "column not in export"
        out["score"] = ACCEPTABLE
        return out
    entry = pd.to_datetime(frame["Entry Date"], errors="coerce")
    age = pd.Timestamp(access_date) - pd.Timestamp(f"{last}-12-31")
    lo, hi = cross["emdat_mature_years"]
    shares = {}
    for y in range(int(lo), int(hi) + 1):
        sel = (years == y).to_numpy(dtype=bool) & entry.notna().to_numpy(dtype=bool)
        cutoff = pd.Timestamp(f"{y}-12-31") + age
        shares[str(y)] = round(100 * float((entry[sel] > cutoff).mean()), 1)
    expected = float(np.median(list(shares.values())))
    out.update(
        {
            "entry_date_missing": int(entry.isna().sum()),
            "export_age_of_last_year_days": age.days,
            "late_entry_pct_by_year": shares,
            "expected_missing_pct_last_year": round(expected, 1),
            "score": grade(expected, cross["emdat_late_entry_pct"]),
        }
    )
    return out


# ---------------------------------------------------------------- ERA5 extremes
def _top_cells(
    values: np.ndarray,
    when: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    times: list[str],
    *,
    largest: bool,
) -> list[dict[str, Any]]:
    """Return the ``N_EXAMPLES`` most extreme cells (longitude converted to -180..180)."""
    order = np.argsort(values, axis=None)
    picks = order[::-1][:N_EXAMPLES] if largest else order[:N_EXAMPLES]
    cells = []
    for flat in picks:
        i, j = np.unravel_index(flat, values.shape)
        lon_e = float(lon[j]) - 360 if lon[j] > 180 else float(lon[j])
        cells.append(
            {
                "lat": float(lat[i]),
                "lon": round(lon_e, 2),
                "month": times[int(when[i, j])],
                "value": float(values[i, j]),
            }
        )
    return cells


def _update(state: dict[str, Any], grid: np.ndarray, index: int) -> None:
    """Keep the per-cell running maximum/minimum and the month index at which it occurred."""
    for kind, better in (("max", np.greater), ("min", np.less)):
        if kind not in state:
            state[kind] = grid.astype("float64").copy()
            state[f"{kind}_t"] = np.full(grid.shape, index)
            continue
        hit = better(grid, state[kind])
        state[kind][hit] = grid[hit]
        state[f"{kind}_t"][hit] = index


def era5_extremes(entries: list[ManifestEntry], root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Locate the grid cells with the most extreme monthly t2m (max/min) and tp (max)."""
    state: dict[str, dict[str, Any]] = {}
    for entry in sorted(entries, key=lambda e: e.file):
        logger.info("ERA5 extremes: %s", entry.file)
        with zipfile.ZipFile(root / entry.file) as archive, tempfile.TemporaryDirectory() as tmp:
            for info in archive.infolist():
                with xr.open_dataset(Path(archive.extract(info, tmp))) as ds:
                    stamps = pd.DatetimeIndex(ds["valid_time"].to_numpy()).strftime("%Y-%m")
                    for var in ("t2m", "tp"):
                        if var not in ds:
                            continue
                        s = state.setdefault(var, {"times": []})
                        s["lat"], s["lon"] = ds["latitude"].to_numpy(), ds["longitude"].to_numpy()
                        offset = len(s["times"])
                        s["times"].extend(stamps.tolist())
                        for k in range(0, len(stamps), 12):
                            block = ds[var].isel(valid_time=slice(k, k + 12)).to_numpy()
                            for t, grid in enumerate(block):
                                _update(s, grid, offset + k + t)
    out: dict[str, Any] = {}
    for var, s in state.items():
        args = (s["lat"], s["lon"], s["times"])
        out[var] = {"highest": _top_cells(s["max"], s["max_t"], *args, largest=True)}
        if var == "t2m":
            out[var]["lowest"] = _top_cells(s["min"], s["min_t"], *args, largest=False)
    return out


# ---------------------------------------------------------------- orchestration
def run(
    manifest: list[ManifestEntry], root: Path, cfg: dict[str, Any], skip_era5: bool
) -> dict[str, Any]:
    """Run every Step 3.2b check and return the evidence."""
    rules = load_rules()
    cross = rules["crosschecks"]
    historical = {str(k): str(v) for k, v in cross["historical_codes"].items()}
    table = country_table()
    read = {
        name: {t.table: t for t in read_dataset(name, manifest, root)}
        for name in (
            "wdi",
            "owid_co2",
            "faostat_crops_livestock",
            "emdat",
            "nd_gain",
            "berkeley_country_monthly",
            "noaa_co2_mlo_monthly",
        )
    }
    classes = read["wdi"]["country_classification"].frame
    has_region = classes["Region"].astype("string").fillna("").str.strip().ne("")
    reference = set(classes.loc[has_region, "Country Code"])
    out: dict[str, Any] = {
        "reference": {"source": "WDICountry.csv rows with a Region", "n": len(reference)}
    }

    # consistency
    fao_rule = rules["tables"]["faostat_crops_livestock/production"]
    fao_table = read["faostat_crops_livestock"]["production"]
    fao, _ = select_rows(fao_table, fao_rule, {}, cfg)
    consistency: dict[str, Any] = {"faostat": faostat_consistency(fao, cross)}
    consistency["noaa_co2"] = noaa_ancillary(read["noaa_co2_mlo_monthly"]["monthly"].frame)
    owid = read["owid_co2"]["country_year"].frame
    owid = owid[~aggregate_mask("owid", owid, {}).to_numpy(dtype=bool)]
    consistency["owid_code_name"] = one_to_one(owid, "iso_code", "country")
    wdi = read["wdi"]["indicators"].frame
    consistency["wdi_code_name"] = one_to_one(wdi, "Country Code", "Country Name")
    consistency["wdi_indicator_name"] = one_to_one(wdi, "Indicator Code", "Indicator Name")
    emdat = read["emdat"]["events"].frame
    consistency["emdat_code_name"] = one_to_one(emdat, "ISO", "Country")
    for name, spec in cross["required_values"].items():
        dataset, tab = name.split("/")
        consistency[f"required_values {name}"] = required_values(
            read[dataset][tab].frame, spec, cfg
        )
    access = max(e.access_date for e in manifest if e.dataset == "emdat")
    consistency["emdat_reporting_lag"] = emdat_lag(emdat, access, cross)
    out["consistency"] = consistency

    # integrity
    integ: dict[str, Any] = {}
    data_codes = set(wdi["Country Code"])
    integ["wdi_codes_without_classification"] = sorted(data_codes - set(classes["Country Code"]))
    wdi_c = wdi[~aggregate_mask("wdi", wdi, read["wdi"]).to_numpy(dtype=bool)]
    integ["wdi"] = classify_codes(
        wdi_c["Country Code"], wdi_c["Country Name"], reference, table, historical, cross
    )
    integ["owid_co2"] = classify_codes(
        owid["iso_code"], owid["country"], reference, table, historical, cross
    )
    areas = fao[["Area Code", "Area Code (M49)", "Area"]].astype("string").drop_duplicates()
    mapped = fao_to_iso3(areas, table)
    unmapped = mapped[mapped["iso3"].isna()]
    dup = mapped.dropna(subset=["iso3"]).groupby("iso3")["Area"].unique()
    integ["faostat_mapping"] = {
        "n_areas": len(mapped),
        "by_source": mapped["iso3_source"].value_counts().to_dict(),
        "unmapped_areas": [
            f"{r['Area Code']} {r['Area']} (M49 {r['Area Code (M49)']})"
            for _, r in unmapped.iterrows()
        ],
        "m49_and_fao_code_disagree": mapped.loc[mapped["sources_disagree"], "Area"].tolist(),
        "iso3_from_several_areas": {k: list(v) for k, v in dup.items() if len(v) > 1},
    }
    known = mapped.dropna(subset=["iso3"])
    integ["faostat"] = classify_codes(
        known["iso3"], known["Area"], reference, table, historical, cross
    )
    integ["faostat"]["historical"] += integ["faostat_mapping"]["unmapped_areas"]
    integ["emdat"] = classify_codes(
        emdat["ISO"], emdat["Country"], reference, table, historical, cross
    )
    nd = read["nd_gain"]
    vul = nd["vulnerability"].frame.dropna(subset=["value"])
    rdy = nd["readiness"].frame.dropna(subset=["value"])
    integ["nd_gain"] = classify_codes(rdy["ISO3"], rdy["Name"], reference, table, historical, cross)
    integ["nd_gain_readiness_without_vulnerability"] = sorted(set(rdy["ISO3"]) - set(vul["ISO3"]))
    bk = read["berkeley_country_monthly"]
    head = bk["file_headers"].frame
    integ["berkeley_country"] = classify_codes(
        head["iso3"], head["region"], reference, table, historical, cross
    )
    monthly = bk["monthly"].frame.dropna(subset=["anomaly"])
    n2020 = monthly[monthly["date"].dt.year == 2020].groupby("iso3").size()
    integ["berkeley_2020"] = {
        "countries_with_2020_data": len(n2020),
        "incomplete_2020": {str(k): int(v) for k, v in n2020[n2020 < 12].items()},
        "header_last_row_counts": head["last_row"].astype(str).value_counts().head(5).to_dict(),
    }
    out["integrity"] = integ
    if not skip_era5:
        out["era5_extremes"] = era5_extremes([e for e in manifest if e.dataset == ERA5], root)
    return out


def render(result: dict[str, Any]) -> str:
    """Render the Step 3.2b evidence as Markdown."""
    lines = [
        "# Consistency and integrity checks (Step 3.2b)",
        "",
        "Generated by `make crosschecks` (rules: `config/quality_rules.yaml`, DEC-065).",
        "",
        "## Scores",
        "",
        "| Check | Score |",
        "|---|---|",
    ]
    for section in ("consistency", "integrity"):
        for name, res in result[section].items():
            if isinstance(res, dict) and "score" in res:
                lines.append(f"| {section}: `{name}` | {res['score']} |")
    for section in ("consistency", "integrity", "era5_extremes"):
        if section not in result:
            continue
        lines += ["", f"## {section.replace('_', ' ').capitalize()}", ""]
        for name, res in result[section].items():
            lines += [
                f"### `{name}`",
                "",
                "```json",
                json.dumps(res, indent=1, default=str),
                "```",
                "",
            ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the Step 3.2b checks and write ``reports/quality/crosschecks.{json,md}``."""
    parser = argparse.ArgumentParser(description="Step 3.2b consistency and integrity checks.")
    parser.add_argument("--skip-era5", action="store_true", help="skip the ERA5 extreme scan")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "quality")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(load_manifest(), PROJECT_ROOT, load_config(), args.skip_era5)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "crosschecks.json").write_text(
        json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    (args.out / "crosschecks.md").write_text(render(result), encoding="utf-8", newline="\n")
    logger.info("wrote %s", args.out / "crosschecks.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
