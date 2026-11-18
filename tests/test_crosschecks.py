"""Step 3.2b tests: consistency, integrity, ERA5 extremes, EM-DAT lag (offline)."""

import zipfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_impact.config import load_config
from climate_impact.provenance import ManifestEntry
from climate_impact.quality import checks, crosschecks

CFG = load_config()
CROSS = checks.load_rules()["crosschecks"]
HIST = {str(k): str(v) for k, v in CROSS["historical_codes"].items()}


@pytest.fixture(scope="module")
def table():
    return crosschecks.country_table()


# ------------------------------------------------------------------ country reference
def test_country_table_survives_the_namibia_trap(table):
    # 'NA' (Namibia's ISO2) must stay text; country_converter's own API fails here under pandas 3
    assert table.loc[table["ISO3"] == "NAM", "ISO2"].item() == "NA"
    assert table.loc[table["FAOcode"] == "351", "ISO3"].item() == "CHN"  # the aggregate trap


def test_fao_mapping_prefers_m49_and_reports_disagreement(table):
    areas = pd.DataFrame(
        {
            "Area Code": ["41", "351", "228", "206", "1"],
            "Area Code (M49)": ["'156", "'159", "'810", "'736", "'004"],
            "Area": ["China, mainland", "China", "USSR", "Sudan (former)", "Mislabelled"],
        }
    )
    out = crosschecks.fao_to_iso3(areas, table).set_index("Area")
    assert out.loc["China, mainland", ["iso3", "iso3_source"]].tolist() == ["CHN", "m49"]
    assert out.loc["China", "iso3_source"] == "fao_code"  # M49 159 is not a country code
    assert out.loc["USSR", "iso3"] == "SUN"
    assert pd.isna(out.loc["Sudan (former)", "iso3"])
    assert bool(out.loc["Mislabelled", "sources_disagree"])  # M49 004 = AFG, FAO code 1 = ARM


def test_classify_codes_separates_territories_history_and_errors(table):
    codes = pd.Series(["KEN", "ABW", "SUN", "YUG", "XYZ", "KEN"])
    names = pd.Series(["Kenya", "Aruba", "USSR", "Yugoslavia", "Typo", "Kenya"])
    out = crosschecks.classify_codes(codes, names, {"KEN", "NOR"}, table, HIST, CROSS)
    assert out["n_codes"] == 5
    assert out["territory_or_other"] == ["ABW (Aruba)"]
    assert out["historical"] == ["SUN (USSR)", "YUG (Yugoslavia)"]
    assert out["invalid"] == ["XYZ (Typo)"]
    assert out["reference_codes_absent"] == ["NOR"]
    assert out["reference_coverage_pct"] == 50.0
    assert out["score"] == "Problematic"  # 1 of 5 codes invalid = 20 %


# ------------------------------------------------------------------ consistency
def fao_rows(rows):
    cols = ["Area Code", "Item Code", "Year", "Element Code", "Unit", "Value", "Flag"]
    frame = pd.DataFrame(rows, columns=cols)
    for c in ("Area Code", "Item Code", "Element Code", "Unit", "Flag"):
        frame[c] = frame[c].astype("category")
    return frame


def test_faostat_yield_identity_units_and_flags():
    rows = [
        ("1", "56", 2000, "5312", "ha", 100.0, "A"),
        ("1", "56", 2000, "5510", "t", 500.0, "A"),
        ("1", "56", 2000, "5412", "kg/ha", 5000.0, "A"),  # 500 t / 100 ha = 5000 kg/ha: exact
        ("2", "56", 2000, "5312", "ha", 100.0, "A"),
        ("2", "56", 2000, "5510", "t", 500.0, "A"),
        ("2", "56", 2000, "5412", "kg/ha", 6000.0, "E"),  # 20 % off
        ("3", "56", 2000, "5312", "ha", np.nan, "M"),  # blank, correctly flagged
        ("3", "56", 2001, "5312", "ha", np.nan, "E"),  # blank without M
    ]
    out = crosschecks.faostat_consistency(fao_rows(rows), CROSS)
    assert out["units_ok"]
    ident = out["yield_identity"]
    assert (ident["n_triples"], ident["n_off"]) == (2, 1)
    assert ident["examples"][0]["computed_kg_ha"] == 5000.0
    assert ident["score"] == "Problematic"  # 50 % of triples off
    assert out["blank_vs_flag"]["blank_without_flag_M"] == 1
    assert out["score"] == "Problematic"


def test_faostat_mixed_units_fail():
    rows = [
        ("1", "56", 2000, "5412", "kg/ha", 1.0, "A"),
        ("1", "56", 2001, "5412", "hg/ha", 1.0, "A"),
    ]
    out = crosschecks.faostat_consistency(fao_rows(rows), CROSS)
    assert not out["units_ok"]
    assert out["units_per_element"]["5412"] == ["hg/ha", "kg/ha"]


def test_noaa_ancillary_disagreement_is_listed():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["1958-03-01", "1958-04-01", "1974-05-01"]),
            "average": [315.7, 317.5, 333.0],
            "deseasonalized": [314.4, 315.2, 331.0],
            "ndays": [np.nan, np.nan, 14.0],
            "sdev": [np.nan, 0.5, np.nan],
            "unc": [np.nan, np.nan, 0.3],
        }
    )
    out = crosschecks.noaa_ancillary(frame)
    assert [m["date"] for m in out["months_disagreeing"]] == ["1958-04-01", "1974-05-01"]
    assert out["co2_missing_in_those_months"] == 0
    assert out["score"] == "Acceptable"


def test_one_to_one():
    renamed = pd.DataFrame({"c": ["SWZ", "SWZ", "KEN"], "n": ["Swaziland", "Eswatini", "Kenya"]})
    assert crosschecks.one_to_one(renamed, "c", "n")["score"] == "Acceptable"
    clash = pd.DataFrame({"c": ["COD", "ZAR"], "n": ["Congo", "Congo"]})
    out = crosschecks.one_to_one(clash, "c", "n")
    assert out["names_with_several_codes"] == {"Congo": ["COD", "ZAR"]}
    assert out["score"] == "Problematic"


def test_required_values_catch_misspelled_config():
    frame = pd.DataFrame({"Disaster Subgroup": ["Hydrological", "Geophysical"]})
    spec = {"Disaster Subgroup": ["Hydrological", "Meteorological"]}
    out = crosschecks.required_values(frame, spec, CFG)
    assert out == {"absent": {"Disaster Subgroup": ["Meteorological"]}, "score": "Problematic"}


def test_emdat_lag_estimates_incompleteness_of_last_year():
    rows = []
    for year in range(2015, 2026):
        for i in range(20):
            late = i < 2  # 10 % of events entered 2 years after the year ends
            entry = pd.Timestamp(f"{year + (2 if late else 0)}-06-30")
            rows.append({"Start Year": year, "Entry Date": entry.strftime("%Y-%m-%d")})
    out = crosschecks.emdat_lag(pd.DataFrame(rows), "2026-11-17", CROSS)
    assert out["last_full_year"] == 2025
    assert out["late_entry_pct_by_year"]["2016"] == 10.0
    assert out["expected_missing_pct_last_year"] == 10.0
    assert out["score"] == "Acceptable"


def test_emdat_lag_without_entry_dates_uses_counts_only():
    frame = pd.DataFrame({"Start Year": [2016] * 10 + [2020] * 12 + [2025] * 5})
    out = crosschecks.emdat_lag(frame, "2026-11-17", CROSS)
    assert out["entry_date"] == "column not in export"
    assert out["events_last_year"] == 5


# ------------------------------------------------------------------ ERA5 extremes
def test_era5_extremes_locate_cells_and_months(tmp_path):
    times = pd.to_datetime(["2020-01-01", "2020-02-01"])
    lat, lon = [10.0, 0.0], [0.0, 270.0]
    t2m = np.full((2, 2, 2), 280.0)
    t2m[1, 0, 1] = 320.0  # hottest: Feb 2020 at 10N, 270E = 90W
    t2m[0, 1, 0] = 200.0  # coldest: Jan 2020 at 0N, 0E
    tp = np.zeros((2, 2, 2))
    tp[0, 1, 1] = 0.1
    paths = []
    for name, var, data in (("a.nc", "t2m", t2m), ("b.nc", "tp", tp)):
        ds = xr.Dataset(
            {var: (("valid_time", "latitude", "longitude"), data)},
            coords={"valid_time": times, "latitude": lat, "longitude": lon},
        )
        ds.to_netcdf(tmp_path / name)
        paths.append(tmp_path / name)
    (tmp_path / "era5").mkdir()
    with zipfile.ZipFile(tmp_path / "era5" / "x.zip", "w") as archive:
        for p in paths:
            archive.write(p, p.name)
    entry = ManifestEntry("era5", "era5/x.zip", "0" * 64, 0, "", "", "2026-11-18", "", "cdsapi")
    out = crosschecks.era5_extremes([entry], tmp_path)
    assert out["t2m"]["highest"][0] == {
        "lat": 10.0,
        "lon": -90.0,
        "month": "2020-02",
        "value": 320.0,
    }
    assert out["t2m"]["lowest"][0] == {"lat": 0.0, "lon": 0.0, "month": "2020-01", "value": 200.0}
    assert out["tp"]["highest"][0]["month"] == "2020-01"
    assert "lowest" not in out["tp"]


def test_crosscheck_rules_are_numbers():
    for key in (
        "integrity_invalid_pct",
        "yield_identity_pct",
        "blank_flag_pct",
        "emdat_late_entry_pct",
    ):
        assert all(isinstance(v, int | float) for v in CROSS[key].values())
    assert set(HIST) >= {"SUN", "YUG", "CSK"}


# ------------------------------------------------------------------ end to end
def fake_tables():
    from climate_impact.ingest.readers import RawTable

    def t(dataset, name, frame, **kw):
        return RawTable(dataset, name, frame, ["f"], **kw)

    years = list(range(2020, 2025))
    wdi = pd.DataFrame(
        [
            (n, c, "Pop", "SP.POP.TOTL", y, 1e6)
            for c, n in (("KEN", "Kenya"), ("WLD", "World"))
            for y in years
        ],
        columns=[
            "Country Name",
            "Country Code",
            "Indicator Name",
            "Indicator Code",
            "year",
            "value",
        ],
    )
    classes = pd.DataFrame(
        {
            "Country Code": ["KEN", "WLD"],
            "Region": ["Sub-Saharan Africa", ""],
            "Income Group": ["Lower middle income", ""],
        }
    )
    owid = pd.DataFrame({"country": "Kenya", "iso_code": "KEN", "year": years, "co2": 1.0})
    fao = pd.DataFrame(
        [
            ("114", "'404", "Kenya", "56", e, u, 2020, v, "A")
            for e, u, v in (("5312", "ha", 10.0), ("5510", "t", 20.0), ("5412", "kg/ha", 2000.0))
        ],
        columns=[
            "Area Code",
            "Area Code (M49)",
            "Area",
            "Item Code",
            "Element Code",
            "Unit",
            "Year",
            "Value",
            "Flag",
        ],
    )
    for c in ("Area Code", "Area Code (M49)", "Area", "Item Code", "Element Code", "Unit", "Flag"):
        fao[c] = fao[c].astype("category")
    emdat = pd.DataFrame(
        {
            "DisNo.": ["2020-1-KEN", "2025-1-KEN"],
            "ISO": "KEN",
            "Country": "Kenya",
            "Disaster Subgroup": ["Hydrological", "Geophysical"],
            "Start Year": [2020, 2025],
        }
    )
    nd = pd.DataFrame({"ISO3": "KEN", "Name": "Kenya", "year": years, "value": 0.5})
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    bk = pd.DataFrame({"iso3": "KEN", "date": dates, "anomaly": 0.1})
    head = pd.DataFrame({"iso3": ["KEN"], "region": ["Kenya"], "last_row": [dates[-1]]})
    co2 = pd.DataFrame(
        {
            "date": dates,
            "average": 410.0,
            "deseasonalized": 410.0,
            "ndays": 20.0,
            "sdev": 0.3,
            "unc": 0.1,
        }
    )
    return {
        "wdi": [t("wdi", "indicators", wdi), t("wdi", "country_classification", classes)],
        "owid_co2": [t("owid_co2", "country_year", owid)],
        "faostat_crops_livestock": [t("faostat_crops_livestock", "production", fao)],
        "emdat": [t("emdat", "events", emdat)],
        "nd_gain": [t("nd_gain", "vulnerability", nd), t("nd_gain", "readiness", nd)],
        "berkeley_country_monthly": [
            t("berkeley_country_monthly", "monthly", bk),
            t("berkeley_country_monthly", "file_headers", head),
        ],
        "noaa_co2_mlo_monthly": [t("noaa_co2_mlo_monthly", "monthly", co2)],
    }


def test_run_end_to_end_on_synthetic_datasets(monkeypatch, tmp_path):
    tables = fake_tables()
    monkeypatch.setattr(crosschecks, "read_dataset", lambda name, manifest, root: tables[name])
    manifest = [ManifestEntry("emdat", "e.xlsx", "0" * 64, 0, "", "", "2026-11-17", "", "manual")]
    result = crosschecks.run(manifest, tmp_path, CFG, skip_era5=True)
    integ = result["integrity"]
    assert result["reference"]["n"] == 1
    assert integ["faostat"]["n_in_reference"] == 1  # Kenya via M49 404
    assert integ["wdi"]["reference_codes_absent"] == []
    assert result["consistency"]["faostat"]["yield_identity"]["n_off"] == 0
    absent = result["consistency"]["required_values emdat/events"]["absent"]
    assert absent == {"Disaster Subgroup": ["Meteorological", "Climatological"]}
    page = crosschecks.render(result)
    assert "| consistency: `faostat` |" in page
