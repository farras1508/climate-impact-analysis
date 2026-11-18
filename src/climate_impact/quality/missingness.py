"""Step 3.4: missingness patterns and a formal test of what predicts them (DEC-076).

For every variable as an analysis will use it, a grid of units x years is built over the
reference universe (217 WDI economies), restricted to the years in which each country exists
as a territorially constant unit (DEC-074). The outcome is ``missing`` (1 = no value).

Model (one per variable): binomial GLM (logit link) of ``missing`` on income group, World
Bank region, log population (standardised) and year (per decade), with standard errors
clustered by country, because a country's missingness is serially correlated. Joint Wald
tests per term answer "does missingness depend on this observed characteristic?".

Multiplicity: the terms of one model form a family ("does missingness depend on any observed
covariate?"), so p-values are Holm-adjusted within each model before classification.

Interpretation (Rubin 1976; Little & Rubin 2019):
    * no term significant: MCAR not rejected (it can never be proven);
    * some term significant: not MCAR; MAR given the covariates is plausible;
    * MNAR (missingness caused by the unobserved value itself) cannot be tested from the
      observed data, so it is stated as an assumption, never claimed to be excluded.

GDP per capita is among the tested variables, so income group stands in for it as a
covariate (no circularity). Only *whether* a value exists is used: values are never read.

Usage: ``python -m climate_impact.quality.missingness`` (``make missingness``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from statsmodels.stats.multitest import multipletests

from climate_impact.config import PROJECT_ROOT, load_config
from climate_impact.ingest.readers import RawTable, read_dataset
from climate_impact.provenance import ManifestEntry, load_manifest
from climate_impact.quality.checks import load_rules, select_rows
from climate_impact.quality.crosschecks import country_table, fao_to_iso3

logger = logging.getLogger(__name__)

INCOME_ORDER = ["Low income", "Lower middle income", "Upper middle income", "High income"]
UNCLASSIFIED = "Unclassified"
# Sequential blue ramp (light -> dark) and the neutral grey for "unit does not exist".
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NOT_EXISTING = "#bcbbb6"  # mid neutral grey: distinct from the lightest blue and from missing
TERM_LABELS = {
    "C(region)": "region",
    "C(crop)": "crop",
    "log_pop": "log population",
    "decade": "year",
}


def _label(term: str) -> str:
    """Plain-language name of a model term (``C(income, ...)`` -> ``income group``)."""
    return "income group" if term.startswith("C(income") else TERM_LABELS.get(term, term)


INK, MUTED = "#2b2b2b", "#6b6b6b"


# ---------------------------------------------------------------- universe and grids
def universe(classes: pd.DataFrame) -> pd.DataFrame:
    """Return the reference economies (Region non-blank) with region and income group."""
    region = classes["Region"].astype("string").fillna("").str.strip()
    income = classes["Income Group"].astype("string").fillna("").str.strip()
    out = pd.DataFrame(
        {"iso3": classes["Country Code"].astype("string"), "region": region, "income": income}
    )[region.ne("").to_numpy(dtype=bool)]
    out["income"] = out["income"].replace("", UNCLASSIFIED)
    return out.sort_values("iso3").reset_index(drop=True)


def entry_years(successions: dict[str, Any]) -> dict[str, int]:
    """First year each successor exists as a constant territory: predecessor end + 1."""
    entry: dict[str, int] = {}
    for spec in successions.values():
        for code in spec["successors"]:
            entry[str(code)] = max(entry.get(str(code), 0), int(spec["ended"]) + 1)
    return entry


def country_year_grid(
    countries: pd.DataFrame, window: tuple[int, int], entry: dict[str, int]
) -> pd.DataFrame:
    """Every country x year in the window, minus years before the country exists."""
    years = pd.DataFrame({"year": range(window[0], window[1] + 1)})
    grid = countries.merge(years, how="cross")
    first = grid["iso3"].map(entry).fillna(window[0]).astype(int)
    return grid[grid["year"] >= first].reset_index(drop=True)


def add_log_pop(grid: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Standardised log population (country-year; country median where a year is missing)."""
    pop = population.rename(columns={"Country Code": "iso3"})[["iso3", "year", "value"]]
    out = grid.merge(pop, on=["iso3", "year"], how="left", validate="many_to_one")
    log_pop = np.log(out["value"].where(out["value"] > 0))
    filled = log_pop.fillna(log_pop.groupby(out["iso3"]).transform("median"))
    spread = float(filled.std())
    out["log_pop"] = (filled - filled.mean()) / spread if spread > 0 else filled * 0.0
    return out.drop(columns="value")


# ---------------------------------------------------------------- model
def _odds(log_odds: float, limit: float) -> float | None:
    """Odds ratio, or None where separation makes it unbounded (JSON has no Infinity)."""
    value = float(log_odds)
    return None if not np.isfinite(value) or abs(value) > limit else round(float(np.exp(value)), 3)


def fit_missingness(
    frame: pd.DataFrame,
    terms: list[str],
    separation_abs_coef: float,
    alpha: float,
) -> dict[str, Any]:
    """Logit of ``missing`` on the given terms, country-clustered; joint Wald test per term."""
    data = frame.dropna(subset=["log_pop"]).copy()
    out: dict[str, Any] = {
        "n": len(frame),
        "n_countries": int(frame["iso3"].nunique()),
        "pct_missing": round(100 * float(frame["missing"].mean()), 2) if len(frame) else 0.0,
        "n_model": len(data),
        "n_dropped_no_population": len(frame) - len(data),
    }
    rate = float(data["missing"].mean()) if len(data) else 0.0
    if rate in (0.0, 1.0):
        out.update(estimable=False, reason="no variation in missingness")
        return out
    present = [t for t in terms if t in data or t.startswith("C(")]
    usable = []
    for term in present:
        column = term[2:].split(",")[0].split(")")[0] if term.startswith("C(") else term
        if not data[column].eq(data[column].iloc[0]).all():
            usable.append(term)
    formula = "missing ~ " + " + ".join(usable) if usable else "missing ~ 1"
    groups = pd.factorize(data["iso3"])[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = smf.glm(formula, data=data, family=sm.families.Binomial())
            res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
            tests = res.wald_test_terms(skip_single=False, scalar=True).table
        except (np.linalg.LinAlgError, ValueError) as error:
            out.update(estimable=False, reason=f"model not identifiable: {error}", formula=formula)
            return out
    params, conf = res.params, res.conf_int()
    separation = bool((params.abs() > separation_abs_coef).any()) or any(
        "separation" in str(w.message).lower() for w in caught
    )
    term_p = {
        _label(str(name)): round(float(row["pvalue"]), 4)
        for name, row in tests.iterrows()
        if name != "Intercept"
    }
    # Holm across the terms of one model: the question is "does missingness depend on ANY
    # observed covariate?", a family of tests (unadjusted, 4 terms give ~19 % false alarms).
    holm = multipletests(list(term_p.values()), alpha=alpha, method="holm")[1]
    term_holm = {t: round(float(p), 4) for t, p in zip(term_p, holm, strict=True)}
    significant = sorted(t for t, p in term_holm.items() if p < alpha)
    out.update(
        estimable=True,
        formula=formula,
        separation=separation,
        term_pvalues=term_p,
        term_pvalues_holm=term_holm,
        odds_ratios={
            str(k): {
                "or": _odds(params[k], separation_abs_coef),
                "ci95": [
                    _odds(conf.loc[k, 0], separation_abs_coef),
                    _odds(conf.loc[k, 1], separation_abs_coef),
                ],
            }
            for k in params.index
            if k != "Intercept"
        },
        significant_terms=significant,
        classification=classify(significant),
    )
    return out


def classify(significant: list[str]) -> str:
    """Wording that never claims more than the observed data can show."""
    if not significant:
        return "MCAR not rejected (no dependence on observed covariates detected)"
    return (
        f"not MCAR: missingness depends on {', '.join(significant)}; MAR given these "
        "covariates is plausible; MNAR cannot be tested from observed data"
    )


def describe(frame: pd.DataFrame) -> dict[str, Any]:
    """Missing shares (%) by income group, region and decade; worst countries."""

    def share(by: pd.Series) -> dict[str, float]:
        means = frame["missing"].groupby(by).mean().mul(100).round(1)
        return {str(k): float(v) for k, v in means.to_dict().items()}

    out = {"by_income": share(frame["income"]), "by_region": share(frame["region"])}
    if frame["year"].min() != frame["year"].max():
        out["by_decade"] = share((frame["year"] // 10 * 10).astype(str) + "s")
    worst = frame.groupby("iso3")["missing"].mean().sort_values(ascending=False)
    out["countries_most_missing_pct"] = {
        str(k): round(100 * float(v), 1) for k, v in worst[worst > 0].head(15).to_dict().items()
    }
    return out


# ---------------------------------------------------------------- figure
def heatmap(frame: pd.DataFrame, path: Path, title: str) -> None:
    """Country x year share missing; grey = country-year that does not exist."""
    order = {name: i for i, name in enumerate([*INCOME_ORDER, UNCLASSIFIED])}
    countries = (
        frame[["iso3", "income", "region"]]
        .drop_duplicates("iso3")
        .assign(rank=lambda d: d["income"].map(order))
        .sort_values(["rank", "region", "iso3"])
    )
    cells = frame.pivot_table(index="iso3", columns="year", values="missing", aggfunc="mean")
    cells = cells.reindex(index=countries["iso3"])
    values = np.ma.masked_invalid(cells.to_numpy(dtype=float))
    cmap = ListedColormap(SEQUENTIAL).with_extremes(bad=NOT_EXISTING)
    height = max(4.0, 0.06 * len(cells) + 1.5)
    fig = Figure(figsize=(10, height), dpi=150)
    ax = fig.add_subplot()
    ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    years = list(cells.columns)
    step = 5 if len(years) > 20 else 1
    ax.set_xticks(range(0, len(years), step), [str(y) for y in years[::step]], fontsize=6)
    ax.set_yticks(range(len(cells)), list(cells.index), fontsize=3.5)
    ax.tick_params(colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bounds = countries["income"].to_numpy()
    for i in range(1, len(bounds)):
        if bounds[i] != bounds[i - 1]:
            ax.axhline(i - 0.5, color="white", linewidth=1.5)
    labels, starts = [], []
    for i, group in enumerate(bounds):
        if i == 0 or group != bounds[i - 1]:
            labels.append(str(group))
            starts.append(i)
    for label, start, end in zip(labels, starts, [*starts[1:], len(bounds)], strict=True):
        ax.text(
            len(years) + 0.5,
            (start + end - 1) / 2,
            label,
            fontsize=6,
            color=INK,
            va="center",
            ha="left",
        )
    legend = [
        Patch(facecolor=SEQUENTIAL[0], label="observed"),
        Patch(facecolor=SEQUENTIAL[3], label="partly missing (share of series)"),
        Patch(facecolor=SEQUENTIAL[-1], label="missing"),
        Patch(facecolor=NOT_EXISTING, label="country did not exist (DEC-074)"),
    ]
    ax.legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(0, -0.03),
        ncol=4,
        fontsize=6,
        frameon=False,
        labelcolor=INK,
    )
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    fig.subplots_adjust(left=0.08, right=0.82, top=1 - 0.5 / height, bottom=0.6 / height)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Software": None})


# ---------------------------------------------------------------- assembly
def h3_yield_grid(
    fao: pd.DataFrame,
    countries: pd.DataFrame,
    window: tuple[int, int],
    entry: dict[str, int],
    element: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Grower country-crop pairs x existing years; missing = no yield in that year."""
    codes = fao["Element Code"].astype("string")
    part = fao[codes.eq(str(element)).fillna(False).to_numpy(dtype=bool)]
    areas = part[["Area Code", "Area Code (M49)", "Area"]].astype("string").drop_duplicates()
    iso = fao_to_iso3(areas, country_table()).set_index("Area Code")["iso3"]
    obs = pd.DataFrame(
        {
            "iso3": part["Area Code"].astype("string").map(iso),
            "crop": part["Item Code"].astype("string"),
            "year": part["Year"].astype(int),
            "has": part["Value"].notna().to_numpy(dtype=bool),
        }
    )
    obs = obs[obs["has"] & obs["year"].between(*window)].drop(columns="has").drop_duplicates()
    grid = country_year_grid(countries, window, entry)
    obs = obs[obs["iso3"].isin(set(grid["iso3"]))]
    growers = obs[["iso3", "crop"]].drop_duplicates()
    full = grid.merge(growers, on="iso3")
    full = full.merge(obs.assign(observed=1), on=["iso3", "crop", "year"], how="left")
    full["missing"] = full["observed"].isna().astype(int)
    span = obs.groupby(["iso3", "crop"])["year"].agg(["min", "max"])
    full = full.join(span, on=["iso3", "crop"])
    kind = np.select(
        [full["missing"].eq(0), full["year"] < full["min"], full["year"] > full["max"]],
        ["observed", "leading", "trailing"],
        "internal",
    )
    counts = {str(k): int(v) for k, v in pd.Series(kind).value_counts().items()}
    return full.drop(columns=["observed", "min", "max"]), counts


def run(
    manifest: list[ManifestEntry], root: Path, cfg: dict[str, Any], figures: Path
) -> dict[str, Any]:
    """Build every grid, describe and model its missingness, draw the heatmaps."""
    rules = load_rules()
    spec = rules["missingness"]
    alpha = float(cfg["statistics"]["alpha"])
    sep = float(spec["separation_abs_coef"])
    cache: dict[str, dict[str, RawTable]] = {}

    def table(name: str, tab: str) -> RawTable:
        if name not in cache:
            cache[name] = {t.table: t for t in read_dataset(name, manifest, root)}
        return cache[name][tab]

    countries = universe(table("wdi", "country_classification").frame)
    entry = entry_years(rules["domain"]["successions"])
    wdi = table("wdi", "indicators").frame
    population = wdi[wdi["Indicator Code"] == "SP.POP.TOTL"]
    panel_terms = [f"C(income, Treatment('{INCOME_ORDER[-1]}'))", "C(region)", "log_pop", "decade"]
    cross_terms = [f"C(income, Treatment('{INCOME_ORDER[-1]}'))", "C(region)", "log_pop"]
    out: dict[str, Any] = {"universe": {"n_countries": len(countries)}, "variables": {}}

    for name, var in spec["variables"].items():
        logger.info("missingness: %s", name)
        result: dict[str, Any] = {"definition": var}
        if var["source"] == "faostat":
            fao_rule = rules["tables"]["faostat_crops_livestock/production"]
            fao, _ = select_rows(table("faostat_crops_livestock", "production"), fao_rule, {}, cfg)
            window = (int(var["window"][0]), int(var["window"][1]))
            grid, kinds = h3_yield_grid(fao, countries, window, entry, int(var["element"]))
            result["missing_by_position"] = kinds
            terms = [*panel_terms, "C(crop)"]
        elif "indicator" in var and "min_valid" not in var:
            window = (int(var["window"][0]), int(var["window"][1]))
            grid = country_year_grid(countries, window, entry)
            values = wdi[wdi["Indicator Code"] == var["indicator"]].rename(
                columns={"Country Code": "iso3"}
            )[["iso3", "year", "value"]]
            grid = grid.merge(values, on=["iso3", "year"], how="left", validate="one_to_one")
            grid["missing"] = grid["value"].isna().astype(int)
            grid = grid.drop(columns="value")
            terms = panel_terms
        else:
            year = int(var.get("year", var.get("window", [0, 0])[1]))
            grid = country_year_grid(countries, (year, year), entry)
            if var["source"] == "nd_gain":
                nd = table("nd_gain", var["table"]).frame
                have = set(nd.loc[(nd["year"] == year) & nd["value"].notna(), "ISO3"])
            else:
                lo, hi = int(var["window"][0]), int(var["window"][1])
                vals = wdi[
                    (wdi["Indicator Code"] == var["indicator"]) & wdi["year"].between(lo, hi)
                ]
                n_valid = vals.dropna(subset=["value"]).groupby("Country Code").size()
                have = set(n_valid[n_valid >= int(var["min_valid"])].index)
            grid["missing"] = (~grid["iso3"].isin(have)).astype(int)
            terms = cross_terms
        grid = add_log_pop(grid, population)
        grid["decade"] = (grid["year"] - 2000) / 10
        result["describe"] = describe(grid)
        result["model"] = fit_missingness(grid, terms, sep, alpha)
        if grid["year"].min() != grid["year"].max():
            fig = figures / f"missingness_{name}.png"
            heatmap(grid, fig, f"Missing values: {name} (Step 3.4)")
            result["figure"] = (
                fig.relative_to(PROJECT_ROOT).as_posix()
                if fig.is_relative_to(PROJECT_ROOT)
                else fig.as_posix()
            )
        out["variables"][name] = result
    return out


def render(result: dict[str, Any]) -> str:
    """Markdown summary table plus the full evidence per variable."""
    lines = [
        "# Missingness patterns (Step 3.4)",
        "",
        "Generated by `make missingness` (rules: `config/quality_rules.yaml`, DEC-076).",
        f"Universe: {result['universe']['n_countries']} WDI economies; years before a country",
        "exists as a constant territory are excluded (DEC-074). MNAR is never testable from",
        "observed data and is stated as an assumption.",
        "",
        "| Variable | Rows | Missing % | Significant terms (Holm p < alpha) | Classification |",
        "|---|---|---|---|---|",
    ]
    for name, res in result["variables"].items():
        m = res["model"]
        terms = ", ".join(m.get("significant_terms", [])) or "-"
        label = m.get("classification", m.get("reason", ""))
        if m.get("separation"):
            label += " (separation: some cells have no missing values)"
        lines.append(f"| `{name}` | {m['n']:,} | {m['pct_missing']} | {terms} | {label} |")
    lines.append("")
    for name, res in result["variables"].items():
        lines += [f"## `{name}`", ""]
        if "figure" in res:
            lines += [f"![{name}](../../{res['figure']})", ""]
        lines += [
            "```json",
            json.dumps({k: v for k, v in res.items() if k != "figure"}, indent=1, default=str),
            "```",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run Step 3.4 and write ``reports/quality/missingness.{json,md}`` plus figures."""
    parser = argparse.ArgumentParser(description="Step 3.4 missingness patterns.")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "quality")
    parser.add_argument("--figures", type=Path, default=PROJECT_ROOT / "reports" / "figures")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(load_manifest(), PROJECT_ROOT, load_config(), args.figures)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "missingness.json").write_text(
        json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    (args.out / "missingness.md").write_text(render(result), encoding="utf-8", newline="\n")
    logger.info("wrote %s", args.out / "missingness.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
