"""Step 3.3 domain-check tests (offline, synthetic data with known answers)."""

import numpy as np
import pandas as pd
import pytest

from climate_impact.config import load_config
from climate_impact.quality import checks, domain

CFG = load_config()
DOM = checks.load_rules()["domain"]


def monthly(start, end, value, iso3=None):
    dates = pd.date_range(f"{start}-01-01", f"{end}-12-01", freq="MS")
    frame = pd.DataFrame({"date": dates, "anomaly": value})
    if iso3:
        frame.insert(0, "iso3", iso3)
    return frame


def test_baseline_confirmed_when_series_averages_zero_over_it():
    frame = pd.concat([monthly(1951, 1980, 0.0), monthly(1981, 2020, 0.5)])
    out = domain.baseline_check(frame, (1951, 1980), (1991, 2020), 0.02)
    assert out["mean_over_baseline"] == 0.0
    assert out["baseline_confirmed"]
    assert (out["months_in_rebase_period"], out["rebase_possible"]) == (360, True)
    assert out["rebase_offset"] == 0.5


def test_wrong_declared_baseline_is_detected():
    # data anomalies relative to 1961-1990 but declared 1951-1980: the 1950s are cooler
    frame = pd.concat([monthly(1951, 1960, -0.3), monthly(1961, 1990, 0.0)])
    out = domain.baseline_check(frame, (1951, 1980), (1991, 2020), 0.02)
    assert out["mean_over_baseline"] == pytest.approx(-0.1)
    assert not out["baseline_confirmed"]
    assert not out["rebase_possible"]


def test_country_baselines_list_failures_and_short_records():
    good = monthly(1951, 2020, 0.0, "AAA")
    shifted = monthly(1951, 2020, 0.2, "BBB")
    short = monthly(1951, 2016, 0.0, "CCC")
    out = domain.country_baselines(
        pd.concat([good, shifted, short]), (1951, 1980), (1991, 2020), 0.02
    )
    assert out["n_countries"] == 3
    assert out["baseline_not_confirmed"] == {"BBB": {"mean": 0.2, "months": 360}}
    assert out["rebase_incomplete"] == {"CCC": 312}


def test_oni_by_decade():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["1955-01-01", "1958-01-01", "1961-01-01"]),
            "ANOM": [-1.0, 0.5, 0.2],
        }
    )
    assert domain.oni_by_decade(frame) == {"1950s": -0.25, "1960s": 0.2}


def test_successions_overlap_gap_and_backcasts():
    present = pd.DataFrame(
        [("SUN", y) for y in range(1961, 1992)]
        + [("RUS", y) for y in range(1985, 2026)]  # back-cast 1985-1991, overlap 1985-1991
        + [("EST", y) for y in range(1994, 2026)],  # starts 1994: gap 1992-1993
        columns=["code", "year"],
    )
    spec = {"USSR": {"ended": 1991, "predecessors": ["SUN"], "successors": ["RUS", "EST", "LVA"]}}
    out = domain.successions(present, spec, (1961, 2025))["USSR"]
    assert out["predecessor_years"] == [1961, 1991]
    assert out["predecessor_years_in_window"] == 31
    assert out["successor_first_year"] == {"RUS": 1985, "EST": 1994}
    assert out["overlap_years"] == list(range(1985, 1992))
    assert out["gap_years"] == []  # RUS starts before the predecessor ended
    assert out["successor_backcast_years_in_window"] == {"RUS": 7}


def test_successions_skip_entities_absent_from_a_dataset():
    present = pd.DataFrame({"code": ["KEN"], "year": [2000]})
    assert domain.successions(present, DOM["successions"], (1961, 2025)) == {}


def test_successions_gap_when_successors_start_late():
    present = pd.DataFrame(
        [("FAO:206", y) for y in range(2000, 2012)] + [("SSD", y) for y in range(2014, 2020)],
        columns=["code", "year"],
    )
    spec = {"Sudan": {"ended": 2011, "predecessors": ["FAO:206"], "successors": ["SDN", "SSD"]}}
    out = domain.successions(present, spec, (1961, 2025))["Sudan"]
    assert out["gap_years"] == [2012, 2013]


def test_emdat_reporting_by_decade_without_climate_subgroups():
    frame = pd.DataFrame(
        {
            "Start Year": [1965, 1968, 2001, 2002, 2003],
            "Disaster Subgroup": [
                "Geophysical",
                "Hydrological",
                "Geophysical",
                "Geophysical",
                "Meteorological",
            ],
            "Total Deaths": [10, np.nan, 5, np.nan, 1],
            "Historic": ["Yes", "Yes", "No", "No", "No"],
        }
    )
    out = domain.emdat_reporting(frame, "Geophysical")
    assert out["1960s"] == {
        "events_all": 2.0,
        "events_geophysical": 1.0,
        "pct_deaths_reported": 50.0,
        "pct_historic": 100.0,
    }
    assert out["2000s"]["events_geophysical"] == 2.0
    assert not any("hydro" in k.lower() or "meteo" in k.lower() for k in out["1960s"])  # blinding


def test_faostat_flags_by_decade():
    frame = pd.DataFrame(
        {
            "Element Code": pd.Categorical(["5412"] * 4),
            "Year": [1961, 1962, 2011, 2012],
            "Flag": pd.Categorical(["A", "A", "A", "E"]),
        }
    )
    out = domain.faostat_flags(frame, [5412])
    assert out["5412"]["1960s"] == {"A": 100.0, "E": 0.0}
    assert out["5412"]["2010s"] == {"A": 50.0, "E": 50.0}


def test_domain_rules_are_complete():
    assert set(DOM["baselines"]) <= set(checks.load_rules()["tables"])
    for name, spec in DOM["successions"].items():
        assert isinstance(spec["ended"], int), name
        assert spec["predecessors"], name
        assert spec["successors"], name
    assert DOM["emdat_reporting_subgroup"] == CFG["disasters"]["negative_control_subgroup"]


def test_render_contains_blinding_statement():
    page = domain.render({"baselines": {"x": {"a": 1}}})
    assert "Outcome" in page
    assert "### `x`" in page


def test_run_end_to_end_on_synthetic_datasets(monkeypatch, tmp_path):
    from climate_impact.ingest.readers import RawTable

    def t(name, frame):
        return RawTable("d", name, frame, ["f"])

    glob = pd.concat([monthly(1951, 1980, 0.0), monthly(1981, 2024, 0.4)])
    oni = pd.DataFrame({"date": pd.to_datetime(["1990-01-01"]), "ANOM": [0.1]})
    fao = pd.DataFrame(
        {
            "Area Code": ["228", "185", "5000"],
            "Area Code (M49)": ["'810", "'643", "'001"],
            "Area": ["USSR", "Russian Federation", "World"],
            "Item Code": ["56", "56", "56"],
            "Element Code": ["5412", "5412", "5412"],
            "Year": [1990, 1992, 1992],
            "Value": [3000.0, 3100.0, 5000.0],
            "Flag": ["A", "E", "A"],
        }
    ).astype(
        dict.fromkeys(
            ("Area Code", "Area Code (M49)", "Area", "Item Code", "Element Code", "Flag"),
            "category",
        )
    )
    emdat = pd.DataFrame(
        {
            "ISO": ["SUN", "RUS"],
            "Start Year": [1990, 1995],
            "Disaster Subgroup": ["Geophysical", "Geophysical"],
            "Total Deaths": [1.0, np.nan],
        }
    )
    wdi = pd.DataFrame({"Country Code": ["RUS"], "year": [1985], "value": [1.0]})
    owid = pd.DataFrame({"iso_code": ["RUS"], "year": [1985], "co2": [1.0]})
    tables = {
        "gistemp_v4_global_monthly": [t("monthly", glob)],
        "hadcrut5_global_monthly": [t("monthly", glob)],
        "berkeley_global_monthly": [t("air_over_sea_ice", glob), t("water_under_sea_ice", glob)],
        "berkeley_country_monthly": [t("monthly", monthly(1951, 2020, 0.0, "KEN"))],
        "noaa_oni": [t("monthly", oni)],
        "faostat_crops_livestock": [t("production", fao)],
        "emdat": [t("events", emdat)],
        "wdi": [t("indicators", wdi)],
        "owid_co2": [t("country_year", owid)],
    }
    monkeypatch.setattr(domain, "read_dataset", lambda name, manifest, root: tables[name])
    out = domain.run([], tmp_path, CFG)
    assert out["baselines"]["hadcrut5_global_monthly/monthly"]["baseline_confirmed"] is False
    assert out["baselines"]["gistemp_v4_global_monthly/monthly"]["baseline_confirmed"] is True
    fao_ussr = out["successions"]["faostat"]["USSR"]
    assert fao_ussr["predecessor_years"] == [1990, 1990]
    assert fao_ussr["successor_first_year"] == {"RUS": 1992}
    assert out["successions"]["wdi"]["USSR"]["successor_backcast_years_in_window"] == {"RUS": 1}
    assert out["reporting_effort"]["emdat_by_decade"]["1990s"]["events_geophysical"] == 2.0
    assert "1990s" in out["reporting_effort"]["faostat_flags_by_decade"]["5412"]
