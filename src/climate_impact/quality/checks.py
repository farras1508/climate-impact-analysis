"""Step 3.2 (part 1): score raw tables on timeliness, completeness, validity and uniqueness.

Every check is driven by ``config/quality_rules.yaml``, whose thresholds were fixed before
the first run (DEC-064). Scores are Good / Acceptable / Problematic, each with the number
behind it. Consistency and integrity (dataset-specific checks) are added in Step 3.2b.

Definitions
-----------
Effective end (timeliness, DEC-037)
    ``complete_years``: last year with ``min_valid_months`` non-missing months.
    ``panel_monthly``: last year in which >= 90 % of in-scope entities have complete years.
    ``panel_annual``: last year whose number of reporting entities is >= 90 % of the median
    of the 5 preceding years (computed per ``by`` group; the table takes the earliest).
    ``events``: last full calendar year before the export (manifest access date).
Completeness
    Single series: missing periods / all periods of the window (up to the effective end).
    Panels: *internal* gaps (missing periods between a series' first and last observation)
    / observed span. Leading and trailing absence is not a gap: a country that does not grow
    soybeans, or did not yet exist, is not "missing"; coverage is reported separately.
    Events: share of missing values in the required columns.
Validity
    Values outside the declared plausibility range (or non-numeric) / values checked.
    Scored inside the window, reported for all rows.
Uniqueness
    Rows sharing a primary key; *conflicting* duplicates (same key, different values) and
    null keys are Problematic, exact duplicates Acceptable.

Usage: ``python -m climate_impact.quality.checks [DATASET ...]`` (``make quality``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from climate_impact.config import CONFIG_DIR, PROJECT_ROOT, load_config
from climate_impact.ingest.readers import READERS, RawTable, read_dataset
from climate_impact.provenance import ManifestEntry, load_manifest

logger = logging.getLogger(__name__)

GOOD, ACCEPTABLE, PROBLEMATIC = "Good", "Acceptable", "Problematic"
N_EXAMPLES = 5
ERA5 = "era5_monthly_single_levels"


def load_rules(path: Path | None = None) -> dict[str, Any]:
    """Read ``config/quality_rules.yaml``."""
    path = path or CONFIG_DIR / "quality_rules.yaml"
    rules = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(rules, dict) or "tables" not in rules:
        raise ValueError(f"{path}: expected a mapping with 'tables'")
    return rules


def grade(value: float, thresholds: dict[str, float]) -> str:
    """Map a metric (smaller = better) to Good / Acceptable / Problematic."""
    if value <= thresholds["good"]:
        return GOOD
    if value <= thresholds["acceptable"]:
        return ACCEPTABLE
    return PROBLEMATIC


def resolve_window(spec: str | list[int], cfg: dict[str, Any]) -> tuple[int, int]:
    """Translate a window name (or an explicit ``[start, end]``) into years."""
    end = int(cfg["analysis"]["study_end_year"])
    if isinstance(spec, list):
        return int(spec[0]), int(spec[1])
    hyp = cfg["hypotheses"]
    named = {
        "study": (int(cfg["analysis"]["study_start_year"]), end),
        "h1b": (int(hyp["h1b"]["window"][0]), int(hyp["h1b"]["window"][1])),
        "h2": (int(str(hyp["h2"]["window"][0])[:4]), end),
        "h4_all": (
            min([*cfg["disasters"]["start_year_sensitivity"], hyp["h4"]["primary_start_year"]]),
            end,
        ),
    }
    if spec not in named:
        raise ValueError(f"unknown window {spec!r}; use one of {sorted(named)} or [start, end]")
    return named[spec]


# ---------------------------------------------------------------- row selection
def _token_values(values: str | list[Any], cfg: dict[str, Any]) -> list[str]:
    """Resolve ``@crops`` / ``@faostat_elements`` to the codes in config.yaml (as text)."""
    if values == "@crops":
        values = list(cfg["crops"].values())
    elif values == "@faostat_elements":
        values = list(cfg["faostat"]["elements"].values())
    if not isinstance(values, list):
        raise ValueError(f"unknown value token {values!r}")
    return [str(v) for v in values]


def aggregate_mask(kind: str, frame: pd.DataFrame, siblings: dict[str, RawTable]) -> pd.Series:
    """Flag rows that are regional/income aggregates rather than countries.

    ``owid``: no ISO code or an ``OWID_*`` code. ``faostat``: FAO area codes >= 5000
    (world/regions) and 351 (China incl. Taiwan, Hong Kong, Macao; its parts are reported
    separately). ``wdi``: codes whose ``Region`` is blank in ``WDICountry.csv``.
    """
    if kind == "owid":
        code = frame["iso_code"].astype("string")
        return code.isna() | code.str.startswith("OWID_").fillna(False)
    if kind == "faostat":
        area = pd.to_numeric(frame["Area Code"].astype("string"), errors="coerce").to_numpy()
        return pd.Series((area >= 5000) | (area == 351), index=frame.index)
    if kind == "wdi":
        classes = siblings["country_classification"].frame
        blank = classes["Region"].astype("string").fillna("").str.strip().eq("")
        return frame["Country Code"].isin(set(classes.loc[blank, "Country Code"]))
    raise ValueError(f"unknown aggregate rule {kind!r}")


def select_rows(
    table: RawTable, rule: dict[str, Any], siblings: dict[str, RawTable], cfg: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    """Apply the rule's row selection (configured codes, aggregates removed)."""
    frame = table.frame
    notes = []
    for col, values in rule.get("include_values", {}).items():
        keep = frame[col].astype("string").isin(_token_values(values, cfg))
        frame = frame[keep.to_numpy(dtype=bool)]
        notes.append(f"kept {col} in {_token_values(values, cfg)}: {len(frame):,} rows")
    if "aggregates" in rule:
        agg = aggregate_mask(rule["aggregates"], frame, siblings).to_numpy(dtype=bool)
        notes.append(f"excluded {int(agg.sum()):,} aggregate rows ({rule['aggregates']})")
        frame = frame[~agg]
    return frame, notes


def _years(frame: pd.DataFrame, table: RawTable) -> pd.Series:
    col = frame[table.time_col]
    if table.time_freq == "M":
        return pd.Series(col.dt.year, index=frame.index)
    return pd.Series(pd.to_numeric(col, errors="coerce"), index=frame.index)


def _periods(frame: pd.DataFrame, table: RawTable) -> pd.Series:
    """Integer period index: months since year 0 (monthly) or the year (annual)."""
    years = _years(frame, table)
    if table.time_freq == "M":
        return pd.Series(years * 12 + frame[table.time_col].dt.month - 1, index=frame.index)
    return years


def _label(period: int, freq: str | None) -> str:
    return f"{period // 12}-{period % 12 + 1:02d}" if freq == "M" else str(period)


# ---------------------------------------------------------------- timeliness
def _counts(series: pd.Series) -> dict[int, float]:
    """Series indexed by year -> plain ``{year: value}`` (keeps mypy and JSON simple)."""
    return {int(k): float(v) for k, v in zip(series.index.tolist(), series.tolist(), strict=True)}


def _panel_annual(
    frame: pd.DataFrame, table: RawTable, value: str, share: float, ref_years: int
) -> tuple[int, dict[str, Any]]:
    valid = frame[value].notna().to_numpy(dtype=bool)
    years = _years(frame, table)[valid]
    entities = frame.loc[valid, table.entity_col]
    n = _counts(entities.groupby(years.to_numpy()).nunique().sort_index())
    ok_years, detail = [], {}
    for year, count in n.items():
        prior = [c for y, c in n.items() if year - ref_years <= y < year]
        ref = float(np.median(prior)) if len(prior) >= 3 else None
        ok = ref is None or count >= share * ref
        if ok:
            ok_years.append(year)
        detail[str(year)] = {"n": int(count), "ref": ref, "ok": bool(ok)}
    last = dict(list(detail.items())[-6:])
    return max(ok_years), last


def effective_end(
    frame: pd.DataFrame,
    table: RawTable,
    rule: dict[str, Any],
    value: str,
    cfg: dict[str, Any],
    scoring: dict[str, Any],
    access_year: int,
) -> dict[str, Any]:
    """Apply the DEC-037 effective-coverage rule configured for this table."""
    spec = rule["effective_end"]
    method = spec["method"]
    min_months = int(cfg["analysis"]["min_valid_months"])
    share = float(scoring["panel_reporting_share"])
    out: dict[str, Any] = {"method": method}
    if method == "complete_years":
        counts = _counts(frame[value].notna().groupby(_years(frame, table).to_numpy()).sum())
        out["effective_end"] = max(y for y, c in counts.items() if c >= min_months)
        out["months_per_year_last"] = {str(y): int(c) for y, c in list(counts.items())[-4:]}
    elif method == "panel_monthly":
        valid = frame[value].notna().to_numpy(dtype=bool)
        sub = pd.DataFrame(
            {"e": frame.loc[valid, table.entity_col], "y": _years(frame, table)[valid]}
        )
        per = sub.groupby(["e", "y"]).size()
        scope = set(sub.loc[sub["y"] == spec["scope_year"], "e"])
        full = per[per >= min_months].reset_index()
        full = full[full["e"].isin(scope)]
        shares = _counts(full.groupby("y")["e"].nunique() / max(len(scope), 1))
        out["effective_end"] = max(y for y, s in shares.items() if s >= share)
        out["n_in_scope"] = len(scope)
        out["n_out_of_scope"] = int(sub["e"].nunique() - len(scope))
        out["complete_share_last"] = {str(y): round(s, 3) for y, s in list(shares.items())[-6:]}
    elif method == "panel_annual":
        ref_years = int(scoring["panel_reference_years"])
        if "by" in spec:
            ends, groups = {}, {}
            for code, part in frame.groupby(spec["by"], observed=True):
                ends[str(code)], groups[str(code)] = _panel_annual(
                    part, table, value, share, ref_years
                )
            out["effective_end"] = min(ends.values())
            out["by"] = {"column": spec["by"], "effective_end": ends, "reporting": groups}
        else:
            out["effective_end"], out["reporting"] = _panel_annual(
                frame, table, value, share, ref_years
            )
    elif method == "events":
        out["effective_end"] = access_year - 1
        years = _years(frame, table)
        per_year = _counts(years.value_counts().sort_index())
        out["access_year"] = access_year
        out["rows_per_year_last"] = {str(y): int(c) for y, c in list(per_year.items())[-6:]}
    else:
        raise ValueError(f"unknown effective_end method {method!r}")
    study_end = int(cfg["analysis"]["study_end_year"])
    out["years_short"] = max(0, study_end - out["effective_end"])
    out["score"] = grade(out["years_short"], scoring["timeliness_years_short"])
    return out


# ---------------------------------------------------------------- completeness
def completeness(
    frame: pd.DataFrame,
    table: RawTable,
    rule: dict[str, Any],
    value: str,
    window: tuple[int, int],
    end: int,
    scoring: dict[str, Any],
) -> dict[str, Any]:
    """Missing share inside ``[window start, min(window end, effective end)]``."""
    start, stop = window[0], min(window[1], end)
    in_window = _years(frame, table).between(start, stop).to_numpy(dtype=bool)
    spec = rule.get("completeness", {})
    out: dict[str, Any] = {"assessed": f"{start}..{stop}"}
    if spec.get("method") == "events":
        sub = frame[in_window]
        pct = {c: round(100 * float(sub[c].isna().mean()), 2) for c in spec["columns"]}
        out.update(method="events", rows=len(sub), pct_missing=pct, metric=max(pct.values()))
    else:
        periods = _periods(frame, table)
        valid = frame[value].notna().to_numpy(dtype=bool) & in_window
        monthly = table.time_freq == "M"
        first = start * 12 if monthly else start
        last = stop * 12 + 11 if monthly else stop
        series = [c for c in [table.entity_col, *rule.get("series", [])] if c]
        if not series:
            observed = set(periods[valid].astype(int))
            missing = [p for p in range(first, last + 1) if p not in observed]
            expected = last - first + 1
            out.update(
                method="single_series",
                expected_periods=expected,
                missing_periods=len(missing),
                missing_examples=[_label(p, table.time_freq) for p in missing[:12]],
                metric=round(100 * len(missing) / expected, 3),
            )
        else:
            obs = frame.loc[valid, series].copy()
            obs["_p"] = periods[valid].astype(int)
            obs = obs.drop_duplicates()
            agg = obs.groupby(series, observed=True)["_p"].agg(["nunique", "min", "max"])
            span = agg["max"] - agg["min"] + 1
            gaps = span - agg["nunique"]
            full = (agg["min"] == first) & (agg["max"] == last) & (gaps == 0)
            worst = gaps[gaps > 0].sort_values(ascending=False).head(N_EXAMPLES)
            out.update(
                method="panel_internal_gaps",
                n_series=len(agg),
                n_series_with_gaps=int((gaps > 0).sum()),
                pct_series_full_window=round(100 * float(full.mean()), 1),
                gap_periods=int(gaps.sum()),
                span_periods=int(span.sum()),
                worst_series=[
                    {
                        "series": [str(k) for k in (key if isinstance(key, tuple) else (key,))],
                        "gaps": int(g),
                    }
                    for key, g in zip(worst.index.tolist(), worst.tolist(), strict=True)
                ],
                metric=round(100 * float(gaps.sum()) / max(int(span.sum()), 1), 3),
            )
    out["score"] = grade(out["metric"], scoring["completeness_gap_pct"])
    return out


# ---------------------------------------------------------------- validity
def _bound(value: object, label: str) -> float | None:
    """Return a range bound as float (``None`` = unbounded); reject non-numbers."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"range bound {value!r} for {label} is not a number (YAML: write 2.0e+9)")
    return float(value)


def _range_checks(
    frame: pd.DataFrame, rule: dict[str, Any]
) -> list[tuple[str, str, float | None, float | None, np.ndarray]]:
    """List (label, column, low, high, row selector) for every configured range."""
    checks = []
    for col, (low, high) in rule.get("ranges", {}).items():
        if col not in frame:
            raise ValueError(f"range configured for missing column {col!r}")
        checks.append(
            (col, col, _bound(low, col), _bound(high, col), np.ones(len(frame), dtype=bool))
        )
    by_spec = rule.get("ranges_by")
    if by_spec:
        codes = frame[by_spec["by"]].astype("string")
        for code, (low, high) in by_spec["ranges"].items():
            sel = codes.eq(str(code)).fillna(False).to_numpy(dtype=bool)
            label = f"{by_spec['column']} [{by_spec['by']}={code}]"
            checks.append((label, by_spec["column"], _bound(low, label), _bound(high, label), sel))
    return checks


def validity(
    frame: pd.DataFrame,
    table: RawTable,
    rule: dict[str, Any],
    window: tuple[int, int],
    scoring: dict[str, Any],
) -> dict[str, Any]:
    """Count values outside plausibility ranges (all rows; scored inside the window)."""
    in_window = _years(frame, table).between(*window).to_numpy(dtype=bool)
    id_cols = [c for c in dict.fromkeys([*table.key, table.time_col]) if c and c in frame]
    details, checked_w, bad_w = [], 0, 0
    for label, col, low, high, sel in _range_checks(frame, rule):
        raw = frame.loc[sel, col]
        values = pd.to_numeric(raw, errors="coerce")
        non_numeric = raw.notna() & values.isna()
        too_low = values < low if low is not None else pd.Series(False, index=values.index)
        too_high = values > high if high is not None else pd.Series(False, index=values.index)
        bad = (too_low | too_high | non_numeric).to_numpy(dtype=bool)
        present = raw.notna().to_numpy(dtype=bool)
        win = in_window[sel]
        checked_w += int((present & win).sum())
        bad_w += int((bad & win).sum())
        distance = pd.concat(
            [
                (low - values) if low is not None else values * 0,
                (values - high) if high is not None else values * 0,
            ],
            axis=1,
        ).max(axis=1)
        worst = distance[bad].sort_values(ascending=False).head(N_EXAMPLES).index
        examples = frame.loc[worst, [*id_cols, col]].astype("string").to_dict("records")
        details.append(
            {
                "check": label,
                "range": [low, high],
                "n_checked": int(present.sum()),
                "n_out": int(bad.sum()),
                "n_out_in_window": int((bad & win).sum()),
                "n_non_numeric": int(non_numeric.sum()),
                "min": None if values.dropna().empty else float(values.min()),
                "max": None if values.dropna().empty else float(values.max()),
                "examples": examples,
            }
        )
    metric = round(100 * bad_w / checked_w, 4) if checked_w else 0.0
    return {
        "checks": details,
        "n_checked_in_window": checked_w,
        "n_out_in_window": bad_w,
        "metric": metric,
        "score": grade(metric, scoring["validity_out_of_range_pct"]),
    }


# ---------------------------------------------------------------- uniqueness
def uniqueness(frame: pd.DataFrame, key: list[str], value_cols: list[str]) -> dict[str, Any]:
    """Primary-key duplicates, split into exact and conflicting, plus null keys."""
    null_keys = int(frame[key].isna().any(axis=1).sum())
    dup = frame.duplicated(subset=key, keep=False).to_numpy(dtype=bool)
    exact = int(frame.duplicated(keep=False).sum())
    conflicting = 0
    examples: list[dict[str, Any]] = []
    if dup.any():
        rows = frame[dup]
        cols = [c for c in frame.columns if c not in key] or key
        varies = rows.groupby(key, observed=True, dropna=False)[cols].nunique(dropna=False)
        bad = varies[(varies > 1).any(axis=1)]
        conflicting = len(bad)
        head = bad.head(N_EXAMPLES).reset_index()[key].astype("string")
        examples = [{str(k): v for k, v in row.items()} for row in head.to_dict("records")]
    if conflicting or null_keys:
        score = PROBLEMATIC
    elif dup.any():
        score = ACCEPTABLE
    else:
        score = GOOD
    return {
        "key": key,
        "rows": len(frame),
        "rows_with_null_key": null_keys,
        "rows_in_duplicate_keys": int(dup.sum()),
        "exact_duplicate_rows": exact,
        "conflicting_keys": conflicting,
        "conflicting_examples": examples,
        "value_cols_compared": value_cols,
        "score": score,
    }


# ---------------------------------------------------------------- per table
def assess_table(
    table: RawTable,
    rule: dict[str, Any],
    siblings: dict[str, RawTable],
    cfg: dict[str, Any],
    scoring: dict[str, Any],
    access_year: int,
) -> dict[str, Any]:
    """Run the four part-1 dimensions on one table."""
    frame, notes = select_rows(table, rule, siblings, cfg)
    result: dict[str, Any] = {"rows_assessed": len(frame), "selection": notes}
    result["uniqueness"] = uniqueness(frame, table.key, table.value_cols)
    if rule.get("uniqueness_only"):
        return result
    value = rule.get("value", table.value_cols[0])
    window = resolve_window(rule["window"], cfg)
    result["window"] = list(window)
    result["timeliness"] = effective_end(frame, table, rule, value, cfg, scoring, access_year)
    end = result["timeliness"]["effective_end"]
    result["completeness"] = completeness(frame, table, rule, value, window, end, scoring)
    result["validity"] = validity(frame, table, rule, window, scoring)
    return result


def assess_era5(
    profile: dict[str, Any], rules: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    """Score ERA5 from the Step 3.1 profile (coverage, NaN, extremes) without a re-scan."""
    era5 = profile["datasets"][ERA5]["era5"]
    scoring = rules["scoring"]
    study_end = int(cfg["analysis"]["study_end_year"])
    out: dict[str, Any] = {}
    for var, (low, high) in rules["era5"]["ranges"].items():
        cov = era5["coverage"][var]
        stats = [
            m["variables"][var]
            for c in era5["chunks"]
            for m in c["members"]
            if var in m["variables"]
        ]
        n_nan = sum(s["n_nan"] for s in stats)
        n_values = sum(s["n_values"] for s in stats)
        vmin, vmax = min(s["min"] for s in stats), max(s["max"] for s in stats)
        last_year = (
            int(cov["last"][:4]) if cov["last"].endswith("-12") else int(cov["last"][:4]) - 1
        )
        short = max(0, study_end - last_year)
        gap = (
            100 * (len(cov["missing_months"]) + n_nan / max(n_values, 1)) / max(cov["n_months"], 1)
        )
        out[var] = {
            "timeliness": {
                "effective_end": last_year,
                "years_short": short,
                "score": grade(short, scoring["timeliness_years_short"]),
            },
            "completeness": {
                "missing_months": cov["missing_months"],
                "n_nan": n_nan,
                "metric": round(gap, 3),
                "score": grade(gap, scoring["completeness_gap_pct"]),
            },
            "validity": {
                "range": [low, high],
                "min": vmin,
                "max": vmax,
                "score": GOOD
                if (low is None or vmin >= low) and (high is None or vmax <= high)
                else PROBLEMATIC,
            },
            "uniqueness": {
                "duplicated_months": cov["n_duplicated"],
                "score": GOOD if cov["n_duplicated"] == 0 else PROBLEMATIC,
            },
        }
    return out


# ---------------------------------------------------------------- report
DIMENSIONS = ("timeliness", "completeness", "validity", "uniqueness")


def _cell(dim: str, res: dict[str, Any] | None) -> str:
    if not res:
        return "-"
    if dim == "timeliness":
        return f"{res['score']} (end {res['effective_end']})"
    if dim == "completeness":
        return f"{res['score']} ({res['metric']}%)"
    if dim == "validity":
        return f"{res['score']} ({res.get('metric', 0)}% out)" if "metric" in res else res["score"]
    if "conflicting_keys" in res:
        return f"{res['score']} ({res['rows_in_duplicate_keys']:,} dup rows)"
    return str(res["score"])


def render_scorecard(results: dict[str, Any]) -> str:
    """Markdown scorecard: one row per table, then the evidence per table."""
    lines = [
        "# Data-quality scorecard (Step 3.2, part 1)",
        "",
        "Generated by `make quality` from `config/quality_rules.yaml` (thresholds fixed before",
        "the first run, DEC-064). Consistency and integrity follow in Step 3.2b.",
        "",
        "| Dataset / table | Timeliness | Completeness | Validity | Uniqueness |",
        "|---|---|---|---|---|",
    ]
    for name, res in results.items():
        cells = " | ".join(_cell(d, res.get(d)) for d in DIMENSIONS)
        lines.append(f"| `{name}` | {cells} |")
    lines += ["", "## Evidence", ""]
    for name, res in results.items():
        lines += [f"### `{name}`", ""]
        for note in res.get("selection", []):
            lines.append(f"- Selection: {note}")
        t = res.get("timeliness")
        if t:
            extra = {
                k: v
                for k, v in t.items()
                if k not in ("score", "method", "effective_end", "years_short")
            }
            lines.append(
                f"- Timeliness ({t['method']}): effective end {t['effective_end']}, "
                f"{t['years_short']} year(s) short of the study end; {json.dumps(extra)}"
            )
        c = res.get("completeness")
        if c:
            extra = {k: v for k, v in c.items() if k not in ("score", "metric")}
            lines.append(f"- Completeness: {c['metric']}%; {json.dumps(extra)}")
        v = res.get("validity")
        if v and "checks" in v:
            lines.append(
                f"- Validity: {v['n_out_in_window']:,} of {v['n_checked_in_window']:,} "
                "values in window outside range"
            )
            for ch in v["checks"]:
                if ch["n_out"]:
                    lines.append(
                        f"  - `{ch['check']}` {ch['range']}: {ch['n_out']:,} out "
                        f"({ch['n_out_in_window']:,} in window), min {ch['min']}, max {ch['max']}; "
                        f"most extreme: {json.dumps(ch['examples'])}"
                    )
        elif v:
            lines.append(f"- Validity: min {v['min']}, max {v['max']}, range {v['range']}")
        u = res.get("uniqueness")
        if u and "key" in u:
            lines.append(
                f"- Uniqueness on {u['key']}: {u['rows_in_duplicate_keys']:,} rows in duplicate "
                f"keys, {u['exact_duplicate_rows']:,} exact duplicates, {u['conflicting_keys']:,} "
                f"conflicting keys, {u['rows_with_null_key']:,} null keys; "
                f"examples {json.dumps(u['conflicting_examples'])}"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the part-1 checks and write ``reports/quality/``."""
    parser = argparse.ArgumentParser(description="Step 3.2 data-quality checks (part 1).")
    parser.add_argument("datasets", nargs="*", help="dataset names (default: all)")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "quality")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg, rules = load_config(), load_rules()
    manifest: list[ManifestEntry] = load_manifest()
    names = args.datasets or list(READERS)
    results: dict[str, Any] = {}
    for name in names:
        access_year = int(max(e.access_date for e in manifest if e.dataset == name)[:4])
        tables = read_dataset(name, manifest)
        siblings = {t.table: t for t in tables}
        for table in tables:
            rule = rules["tables"].get(f"{name}/{table.table}")
            if rule is None:
                continue
            logger.info("%s/%s: checking", name, table.table)
            results[f"{name}/{table.table}"] = assess_table(
                table, rule, siblings, cfg, rules["scoring"], access_year
            )
    profile_path = PROJECT_ROOT / "reports" / "profiling" / "profile.json"
    if not args.datasets and profile_path.exists():
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        for var, res in assess_era5(profile, rules, cfg).items():
            results[f"{ERA5}/{var}"] = res

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "checks.json").write_text(
        json.dumps(results, indent=1, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    (args.out / "scorecard.md").write_text(
        render_scorecard(results) + "\n", encoding="utf-8", newline="\n"
    )
    logger.info("wrote %s (%d table(s))", args.out, len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
