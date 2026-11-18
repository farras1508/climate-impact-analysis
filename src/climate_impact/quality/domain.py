"""Step 3.3: domain-specific checks on how the data were produced (DEC-072).

Baselines
    An anomaly series averages ~0 over its own reference period. Each series' mean over its
    *declared* baseline is therefore an empirical test of the documentation; the 1991-2020
    month count tests whether rebasing (DEC-009) is possible. ONI is reported separately:
    NOAA computes it against centred 30-year base periods that move every 5 years.
Successions (risk RD3)
    Predecessor and successor states per dataset: years present, overlaps (double-counting
    risk), gaps (missing country-years), and successor values dated before the predecessor
    ended (back-casts for territories that were not yet states), inside the study window.
Reporting effort over time
    EM-DAT: events per decade (all and the geophysical placebo subgroup), share with reported
    deaths, share flagged historic. FAOSTAT: flag shares per decade (estimated/imputed values).
Outcome blinding
    Climate-related EM-DAT subgroups and yield *levels* are not tabulated before Phase 6, so
    no data-cleaning choice can be informed by the hypotheses' outcomes.

Usage: ``python -m climate_impact.quality.domain`` (``make domain``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from climate_impact.config import PROJECT_ROOT, load_config
from climate_impact.ingest.readers import RawTable, read_dataset
from climate_impact.provenance import ManifestEntry, load_manifest
from climate_impact.quality.checks import load_rules, select_rows
from climate_impact.quality.crosschecks import country_table, fao_to_iso3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- baselines
def baseline_check(
    frame: pd.DataFrame,
    baseline: tuple[int, int],
    rebase: tuple[int, int],
    tolerance: float,
    value: str = "anomaly",
) -> dict[str, Any]:
    """Mean over the declared baseline (should be ~0) and month counts for rebasing."""
    years = frame["date"].dt.year
    native = frame.loc[years.between(*baseline), value].dropna()
    target = frame.loc[years.between(*rebase), value].dropna()
    expected = 12 * (rebase[1] - rebase[0] + 1)
    mean = float(native.mean()) if len(native) else None
    return {
        "declared_baseline": list(baseline),
        "months_in_baseline": len(native),
        "mean_over_baseline": None if mean is None else round(mean, 4),
        "baseline_confirmed": mean is not None and abs(mean) <= tolerance,
        "months_in_rebase_period": len(target),
        "rebase_possible": len(target) == expected,
        "rebase_offset": None if target.empty else round(float(target.mean()), 4),
    }


def country_baselines(
    frame: pd.DataFrame, baseline: tuple[int, int], rebase: tuple[int, int], tolerance: float
) -> dict[str, Any]:
    """Per-country baseline means (Berkeley): distribution and the countries that fail."""
    rows = {
        str(iso): baseline_check(part, baseline, rebase, tolerance)
        for iso, part in frame.groupby("iso3")
    }
    table = pd.DataFrame(rows).T
    means = pd.to_numeric(table["mean_over_baseline"], errors="coerce").abs()
    failing = table[~table["baseline_confirmed"].astype(bool)]
    no_rebase = table[~table["rebase_possible"].astype(bool)]
    return {
        "n_countries": len(table),
        "abs_mean_quantiles": {
            q: round(float(means.quantile(float(q))), 4) for q in ("0.5", "0.9", "0.99", "1.0")
        },
        "baseline_not_confirmed": {
            k: {"mean": v["mean_over_baseline"], "months": v["months_in_baseline"]}
            for k, v in failing.to_dict("index").items()
        },
        "rebase_incomplete": {
            k: v["months_in_rebase_period"] for k, v in no_rebase.to_dict("index").items()
        },
    }


def oni_by_decade(frame: pd.DataFrame) -> dict[str, float]:
    """Mean ONI per decade: a moving-base index should stay near 0 in every decade."""
    decade = (frame["date"].dt.year // 10) * 10
    means = frame.groupby(decade)["ANOM"].mean()
    return {f"{int(k)}s": round(float(v), 3) for k, v in zip(means.index, means, strict=True)}


# ---------------------------------------------------------------- successions
def successions(
    present: pd.DataFrame, specs: dict[str, Any], window: tuple[int, int]
) -> dict[str, Any]:
    """Predecessor/successor coverage in one dataset.

    ``present`` has columns ``code`` and ``year`` (one row per code-year with data).
    """
    out: dict[str, Any] = {}
    years_by_code = present.groupby("code")["year"].apply(lambda s: sorted(set(s.astype(int))))
    for name, spec in specs.items():
        pred_years = sorted(
            {y for c in spec["predecessors"] for y in years_by_code.get(str(c), [])}
        )
        succ = {
            str(c): years_by_code[str(c)]
            for c in spec["successors"]
            if str(c) in years_by_code.index
        }
        if not pred_years and not succ:
            continue
        succ_years = {y for ys in succ.values() for y in ys}
        overlap = sorted(set(pred_years) & succ_years)
        in_window = [y for y in pred_years if window[0] <= y <= window[1]]
        first_succ = min(min(ys) for ys in succ.values()) if succ else None
        gap = (
            list(range(max(pred_years) + 1, first_succ))
            if pred_years and first_succ and first_succ > max(pred_years) + 1
            else []
        )
        ended = spec.get("ended")
        backcast = {
            c: sum(1 for y in ys if window[0] <= y <= ended)
            for c, ys in succ.items()
            if ended is not None
        }
        out[name] = {
            "ended": ended,
            "predecessor_years": [pred_years[0], pred_years[-1]] if pred_years else None,
            "predecessor_years_in_window": len(in_window),
            "successor_first_year": {c: ys[0] for c, ys in succ.items()},
            "overlap_years": overlap,
            "gap_years": gap,
            "successor_backcast_years_in_window": {c: n for c, n in backcast.items() if n},
        }
    return out


# ---------------------------------------------------------------- reporting effort
def emdat_reporting(frame: pd.DataFrame, placebo: str) -> dict[str, Any]:
    """Events per decade (all, placebo subgroup), share with reported deaths / historic flag."""
    years = pd.to_numeric(frame["Start Year"], errors="coerce")
    decade = (years // 10 * 10).astype("Int64")
    placebo_rows = frame["Disaster Subgroup"].astype("string").eq(placebo).fillna(False)
    table = pd.DataFrame(
        {
            "events_all": frame.groupby(decade).size(),
            f"events_{placebo.lower()}": frame[placebo_rows].groupby(decade[placebo_rows]).size(),
            "pct_deaths_reported": frame.groupby(decade)["Total Deaths"].apply(
                lambda s: round(100 * float(s.notna().mean()), 1)
            ),
        }
    ).fillna(0)
    if "Historic" in frame:
        historic = frame["Historic"].astype("string").str.lower().eq("yes")
        table["pct_historic"] = historic.groupby(decade).mean().mul(100).round(1)
    return {
        f"{k}s": {str(c): float(v) for c, v in row.items()}
        for k, row in table.to_dict("index").items()
    }


def faostat_flags(frame: pd.DataFrame, elements: list[int]) -> dict[str, Any]:
    """Flag shares (%) per decade and element: how much of each decade is official data."""
    out: dict[str, Any] = {}
    codes = frame["Element Code"].astype("string")
    for element in elements:
        part = frame[codes.eq(str(element)).fillna(False).to_numpy(dtype=bool)]
        decade = (part["Year"] // 10 * 10).astype(int)
        shares = (
            pd.crosstab(decade, part["Flag"].astype("string"), normalize="index").mul(100).round(1)
        )
        out[str(element)] = {
            f"{d}s": {str(f): float(v) for f, v in row.items()}
            for d, row in shares.to_dict("index").items()
        }
    return out


# ---------------------------------------------------------------- orchestration
def _present(frame: pd.DataFrame, code: str, year: str, value: str | None) -> pd.DataFrame:
    rows = frame if value is None else frame[frame[value].notna()]
    return pd.DataFrame(
        {"code": rows[code].astype("string"), "year": pd.to_numeric(rows[year], errors="coerce")}
    ).dropna()


def run(manifest: list[ManifestEntry], root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run every Step 3.3 check and return the evidence."""
    rules = load_rules()
    dom = rules["domain"]
    tol = float(dom["baseline_tolerance_degc"])
    rebase = (int(dom["rebase_period"][0]), int(dom["rebase_period"][1]))
    window = (int(cfg["analysis"]["study_start_year"]), int(cfg["analysis"]["study_end_year"]))
    cache: dict[str, dict[str, RawTable]] = {}

    def table(name: str, tab: str) -> RawTable:
        if name not in cache:
            cache[name] = {t.table: t for t in read_dataset(name, manifest, root)}
        return cache[name][tab]

    out: dict[str, Any] = {"baselines": {}}
    for key, base in dom["baselines"].items():
        dataset, tab = key.split("/")
        frame = table(dataset, tab).frame
        span = (int(base[0]), int(base[1]))
        if "iso3" in frame:
            out["baselines"][key] = country_baselines(frame, span, rebase, tol)
        else:
            out["baselines"][key] = baseline_check(frame, span, rebase, tol)
    out["baselines"]["noaa_oni/monthly (moving base, mean by decade)"] = oni_by_decade(
        table("noaa_oni", "monthly").frame
    )

    fao_table = table("faostat_crops_livestock", "production")
    fao_rule = dict(rules["tables"]["faostat_crops_livestock/production"])
    fao_rule.pop("aggregates", None)  # keep predecessor areas; drop only world/regions below
    fao, _ = select_rows(fao_table, fao_rule, {}, cfg)
    area = pd.to_numeric(fao["Area Code"].astype("string"), errors="coerce")
    fao = fao[(area < 5000).to_numpy(dtype=bool) & (area != 351).to_numpy(dtype=bool)]
    areas = fao[["Area Code", "Area Code (M49)", "Area"]].astype("string").drop_duplicates()
    mapped = fao_to_iso3(areas, country_table()).set_index("Area Code")["iso3"]
    fao_codes = fao["Area Code"].astype("string")
    fao_present = pd.DataFrame(
        {"code": fao_codes.map(mapped).fillna("FAO:" + fao_codes), "year": fao["Year"]}
    )[fao["Value"].notna().to_numpy(dtype=bool)]

    emdat = table("emdat", "events").frame
    wdi = table("wdi", "indicators").frame
    owid = table("owid_co2", "country_year").frame
    specs = dom["successions"]
    out["successions"] = {
        "faostat": successions(fao_present, specs, window),
        "emdat": successions(_present(emdat, "ISO", "Start Year", None), specs, window),
        "wdi": successions(_present(wdi, "Country Code", "year", "value"), specs, window),
        "owid_co2": successions(_present(owid, "iso_code", "year", "co2"), specs, window),
    }
    out["reporting_effort"] = {
        "emdat_by_decade": emdat_reporting(emdat, dom["emdat_reporting_subgroup"]),
        "faostat_flags_by_decade": faostat_flags(
            fao, [int(e) for e in dom["faostat_flag_elements"]]
        ),
    }
    return out


def render(result: dict[str, Any]) -> str:
    """Render the Step 3.3 evidence as Markdown."""
    lines = [
        "# Domain-specific checks (Step 3.3)",
        "",
        "Generated by `make domain` (rules: `config/quality_rules.yaml`, DEC-072). Outcome",
        "blinding: climate-related EM-DAT subgroups and yield levels are not tabulated here.",
        "",
    ]
    for section, content in result.items():
        lines += [f"## {section.replace('_', ' ').capitalize()}", ""]
        for name, res in content.items():
            lines += [f"### `{name}`", "", "```json", json.dumps(res, indent=1), "```", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run Step 3.3 and write ``reports/quality/domain.{json,md}``."""
    parser = argparse.ArgumentParser(description="Step 3.3 domain-specific checks.")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "quality")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(load_manifest(), PROJECT_ROOT, load_config())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "domain.json").write_text(
        json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    (args.out / "domain.md").write_text(render(result), encoding="utf-8", newline="\n")
    logger.info("wrote %s", args.out / "domain.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
