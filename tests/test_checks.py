"""Step 3.2 part-1 check tests (offline, synthetic tables with known answers)."""

import numpy as np
import pandas as pd
import pytest

from climate_impact.config import load_config
from climate_impact.ingest.readers import READERS, RawTable
from climate_impact.quality import checks

CFG = load_config()
SCORING = checks.load_rules()["scoring"]


def monthly(values, start="2023-01-01"):
    dates = pd.date_range(start, periods=len(values), freq="MS")
    return RawTable(
        "d",
        "t",
        pd.DataFrame({"date": dates, "v": values}),
        ["f"],
        "date",
        None,
        "M",
        ["v"],
        ["date"],
    )


def panel(rows, key=("iso3", "year")):
    frame = pd.DataFrame(rows, columns=["iso3", "year", "v"])
    return RawTable("d", "t", frame, ["f"], "year", "iso3", "Y", ["v"], list(key))


# ------------------------------------------------------------------ rules and scoring
def test_grade_boundaries_are_inclusive():
    t = {"good": 1.0, "acceptable": 5.0}
    assert [checks.grade(x, t) for x in (0, 1.0, 1.01, 5.0, 5.01)] == [
        "Good",
        "Good",
        "Acceptable",
        "Acceptable",
        "Problematic",
    ]


def test_windows_resolve_from_config():
    assert checks.resolve_window("study", CFG) == (1961, 2025)
    assert checks.resolve_window("h1b", CFG) == (1880, 2025)
    assert checks.resolve_window("h2", CFG) == (1959, 2025)
    assert checks.resolve_window("h4_all", CFG) == (1980, 2025)
    assert checks.resolve_window([1995, 2025], CFG) == (1995, 2025)
    with pytest.raises(ValueError, match="unknown window"):
        checks.resolve_window("h9", CFG)


def test_rules_file_matches_readers_and_windows():
    rules = checks.load_rules()
    for name, rule in rules["tables"].items():
        dataset, _ = name.split("/")
        assert dataset in READERS, name
        if not rule.get("uniqueness_only"):
            checks.resolve_window(rule["window"], CFG)
            assert "effective_end" in rule, name


def test_rules_bounds_are_numbers_not_yaml_strings():
    rules = checks.load_rules()
    for rule in rules["tables"].values():
        bounds = list(rule.get("ranges", {}).values())
        bounds += list(rule.get("ranges_by", {}).get("ranges", {}).values())
        for low, high in bounds:
            for b in (low, high):
                assert b is None or isinstance(b, int | float), (rule, b)


def test_string_bound_is_rejected():
    table = monthly([1.0, 2.0])
    with pytest.raises(TypeError, match="not a number"):
        checks.validity(table.frame, table, {"ranges": {"v": [0, "2.0e9"]}}, (2023, 2023), SCORING)


# ------------------------------------------------------------------ timeliness
def test_complete_years_ignores_partial_final_year():
    table = monthly([0.1] * 12 + [0.2] * 9)  # 2023 complete, 2024 has 9 months
    rule = {"effective_end": {"method": "complete_years"}}
    out = checks.effective_end(table.frame, table, rule, "v", CFG, SCORING, 2026)
    assert out["effective_end"] == 2023
    assert out["years_short"] == 2
    assert out["score"] == "Problematic"


def test_panel_annual_drops_thin_final_year():
    rows = [(c, y, 1.0) for y in range(2015, 2025) for c in "ABCDEFGHIJ"]
    rows += [(c, 2025, 1.0) for c in "ABCDEFGH"]  # 8 of 10 report in 2025 (< 90 %)
    table = panel(rows)
    rule = {"effective_end": {"method": "panel_annual"}}
    out = checks.effective_end(table.frame, table, rule, "v", CFG, SCORING, 2026)
    assert out["effective_end"] == 2024
    assert out["reporting"]["2025"] == {"n": 8, "ref": 10.0, "ok": False}


def test_panel_annual_by_group_takes_earliest_end():
    rows = [(c, y, g) for y in range(2015, 2026) for c in "ABCDEFGHIJ" for g in ("x", "y")]
    frame = pd.DataFrame(rows, columns=["iso3", "year", "g"]).assign(v=1.0)
    frame.loc[(frame["g"] == "y") & (frame["year"] == 2025), "v"] = np.nan
    table = RawTable("d", "t", frame, ["f"], "year", "iso3", "Y", ["v"], ["iso3", "year", "g"])
    rule = {"effective_end": {"method": "panel_annual", "by": "g"}}
    out = checks.effective_end(table.frame, table, rule, "v", CFG, SCORING, 2026)
    assert out["by"]["effective_end"] == {"x": 2025, "y": 2024}
    assert out["effective_end"] == 2024


def test_panel_monthly_scope_excludes_early_ending_files():
    dates = pd.date_range("2016-01-01", "2020-12-01", freq="MS")
    frame = pd.concat(
        [pd.DataFrame({"iso3": c, "date": dates, "v": 1.0}) for c in ("A", "B")]
        + [pd.DataFrame({"iso3": "OLD", "date": dates[:5], "v": 1.0})]  # ends 2016-05
    )
    table = RawTable("d", "t", frame, ["f"], "date", "iso3", "M", ["v"], ["iso3", "date"])
    rule = {"effective_end": {"method": "panel_monthly", "scope_year": 2020}}
    out = checks.effective_end(table.frame, table, rule, "v", CFG, SCORING, 2026)
    assert out["effective_end"] == 2020
    assert (out["n_in_scope"], out["n_out_of_scope"]) == (2, 1)


def test_events_end_is_last_full_year_before_export():
    table = panel([("A", 2025, 1.0), ("A", 2026, 1.0)])
    rule = {"effective_end": {"method": "events"}}
    out = checks.effective_end(table.frame, table, rule, "v", CFG, SCORING, 2026)
    assert out["effective_end"] == 2025
    assert out["score"] == "Good"


# ------------------------------------------------------------------ completeness
def test_single_series_counts_every_missing_month_in_window():
    values = [np.nan, 1.0, 1.0, np.nan] + [1.0] * 8
    table = monthly(values)
    out = checks.completeness(table.frame, table, {}, "v", (2023, 2023), 2023, SCORING)
    assert out["missing_periods"] == 2
    assert out["missing_examples"] == ["2023-01", "2023-04"]
    assert out["metric"] == pytest.approx(100 * 2 / 12, abs=1e-3)


def test_panel_counts_internal_gaps_not_late_starts():
    rows = [("A", y, 1.0) for y in range(2000, 2011) if y != 2005]  # one internal gap
    rows += [("B", y, 1.0) for y in range(2008, 2011)]  # late start: not a gap
    table = panel(rows)
    out = checks.completeness(table.frame, table, {}, "v", (2000, 2010), 2010, SCORING)
    assert out["gap_periods"] == 1
    assert out["span_periods"] == 11 + 3
    assert out["n_series_with_gaps"] == 1
    assert out["worst_series"] == [{"series": ["A"], "gaps": 1}]
    assert out["pct_series_full_window"] == 0.0


def test_completeness_stops_at_effective_end():
    table = monthly([1.0] * 12 + [np.nan] * 12)  # 2024 empty: a timeliness issue only
    out = checks.completeness(table.frame, table, {}, "v", (2023, 2025), 2023, SCORING)
    assert out["assessed"] == "2023..2023"
    assert out["metric"] == 0.0


def test_event_completeness_reports_worst_required_column():
    frame = pd.DataFrame(
        {"ISO": ["A", "B", None, "C"], "Start Year": [2000] * 4, "deaths": [1, None, None, 3]}
    )
    table = RawTable("d", "t", frame, ["f"], "Start Year", "ISO", "Y", ["deaths"], ["ISO"])
    rule = {"completeness": {"method": "events", "columns": ["ISO", "deaths"]}}
    out = checks.completeness(frame, table, rule, "deaths", (2000, 2000), 2000, SCORING)
    assert out["pct_missing"] == {"ISO": 25.0, "deaths": 50.0}
    assert out["score"] == "Problematic"


# ------------------------------------------------------------------ validity
def test_validity_scores_inside_window_and_reports_all_rows():
    table = panel([("A", 1950, -5.0), ("A", 2000, 0.5), ("A", 2001, 9.0), ("A", 2002, 1.0)])
    rule = {"ranges": {"v": [0.0, 2.0]}}
    out = checks.validity(table.frame, table, rule, (2000, 2002), SCORING)
    (check,) = out["checks"]
    assert (check["n_out"], check["n_out_in_window"]) == (2, 1)
    assert out["metric"] == pytest.approx(100 / 3, abs=1e-3)
    assert check["examples"][0]["v"] == "9.0"  # most extreme first
    assert out["score"] == "Problematic"


def test_validity_by_code_and_open_upper_bound():
    frame = pd.DataFrame(
        {"iso3": "A", "year": 2000, "code": ["y", "a", "a"], "v": [0.5, -1.0, 1e12]}
    )
    table = RawTable("d", "t", frame, ["f"], "year", "iso3", "Y", ["v"], ["iso3", "code"])
    rule = {
        "ranges_by": {"column": "v", "by": "code", "ranges": {"y": [1.0, 10.0], "a": [0.0, None]}}
    }
    out = checks.validity(frame, table, rule, (2000, 2000), SCORING)
    by_check = {c["check"]: c["n_out"] for c in out["checks"]}
    assert by_check == {"v [code=y]": 1, "v [code=a]": 1}


def test_validity_counts_non_numeric_text():
    frame = pd.DataFrame({"iso3": "A", "year": [2000, 2001], "v": ["1", "unknown"]})
    table = RawTable("d", "t", frame, ["f"], "year", "iso3", "Y", ["v"], ["iso3", "year"])
    out = checks.validity(frame, table, {"ranges": {"v": [0, 10]}}, (2000, 2001), SCORING)
    assert out["checks"][0]["n_non_numeric"] == 1


# ------------------------------------------------------------------ uniqueness
def test_uniqueness_distinguishes_exact_and_conflicting():
    exact = pd.DataFrame({"k": ["a", "a", "b"], "v": [1, 1, 2]})
    assert checks.uniqueness(exact, ["k"], ["v"])["score"] == "Acceptable"
    conflict = pd.DataFrame({"k": ["a", "a", "b"], "v": [1, 2, 2]})
    out = checks.uniqueness(conflict, ["k"], ["v"])
    assert (out["conflicting_keys"], out["score"]) == (1, "Problematic")
    assert out["conflicting_examples"] == [{"k": "a"}]
    nulls = pd.DataFrame({"k": ["a", None], "v": [1, 2]})
    assert checks.uniqueness(nulls, ["k"], ["v"])["score"] == "Problematic"
    clean = pd.DataFrame({"k": ["a", "b"], "v": [1, 2]})
    assert checks.uniqueness(clean, ["k"], ["v"])["score"] == "Good"


# ------------------------------------------------------------------ selection
def test_aggregate_masks():
    owid = pd.DataFrame({"iso_code": ["KEN", None, "OWID_WRL"]})
    assert checks.aggregate_mask("owid", owid, {}).tolist() == [False, True, True]
    fao = pd.DataFrame({"Area Code": pd.Categorical(["41", "351", "5000", "96"])})
    assert checks.aggregate_mask("faostat", fao, {}).tolist() == [False, True, True, False]
    classes = pd.DataFrame({"Country Code": ["KEN", "WLD"], "Region": ["Sub-Saharan Africa", ""]})
    sib = {"country_classification": RawTable("wdi", "country_classification", classes, ["f"])}
    wdi = pd.DataFrame({"Country Code": ["KEN", "WLD"]})
    assert checks.aggregate_mask("wdi", wdi, sib).tolist() == [False, True]


def test_select_rows_resolves_config_tokens():
    frame = pd.DataFrame(
        {
            "Item Code": pd.Categorical(["56", "999", "15"]),
            "Area Code": pd.Categorical(["41", "41", "5000"]),
        }
    )
    table = RawTable("faostat_crops_livestock", "production", frame, ["f"])
    rule = {"include_values": {"Item Code": "@crops"}, "aggregates": "faostat"}
    out, notes = checks.select_rows(table, rule, {}, CFG)
    assert out["Item Code"].astype(str).tolist() == ["56"]
    assert len(notes) == 2


# ------------------------------------------------------------------ ERA5 from profile
def test_era5_scored_from_profile():
    stats = {"n_values": 100, "n_nan": 0, "min": 200.0, "max": 316.0}
    profile = {
        "datasets": {
            checks.ERA5: {
                "era5": {
                    "coverage": {
                        "t2m": {
                            "first": "1960-01",
                            "last": "2025-12",
                            "n_months": 792,
                            "n_duplicated": 0,
                            "missing_months": [],
                        },
                        "tp": {
                            "first": "1960-01",
                            "last": "2025-11",
                            "n_months": 791,
                            "n_duplicated": 0,
                            "missing_months": [],
                        },
                    },
                    "chunks": [
                        {
                            "members": [
                                {"variables": {"t2m": stats}},
                                {"variables": {"tp": {**stats, "min": -1e-9, "max": 0.1}}},
                            ]
                        }
                    ],
                }
            }
        }
    }
    rules = checks.load_rules()
    out = checks.assess_era5(profile, rules, CFG)
    assert out["t2m"]["validity"]["score"] == "Good"
    assert out["t2m"]["timeliness"]["effective_end"] == 2025
    assert out["tp"]["timeliness"]["effective_end"] == 2024  # 2025 incomplete
    assert out["tp"]["validity"]["score"] == "Problematic"  # negative precipitation


def test_era5_results_render_in_scorecard():
    era5 = {
        "t2m": {
            "first": "1960-01",
            "last": "2025-12",
            "n_months": 792,
            "n_duplicated": 0,
            "missing_months": [],
        }
    }
    stats = {"n_values": 10, "n_nan": 0, "min": 200.0, "max": 300.0}
    profile = {
        "datasets": {
            checks.ERA5: {
                "era5": {"coverage": era5, "chunks": [{"members": [{"variables": {"t2m": stats}}]}]}
            }
        }
    }
    rules = {"scoring": SCORING, "era5": {"ranges": {"t2m": [183.15, 333.15]}}}
    results = {f"{checks.ERA5}/{k}": v for k, v in checks.assess_era5(profile, rules, CFG).items()}
    page = checks.render_scorecard(results)
    assert "Good (end 2025)" in page
