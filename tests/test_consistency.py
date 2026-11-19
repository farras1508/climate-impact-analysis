"""Step 3.5 cross-source consistency tests (offline; synthetic sources with known answers)."""

import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from climate_impact.config import load_config
from climate_impact.ingest.readers import RawTable
from climate_impact.provenance import ManifestEntry
from climate_impact.quality import checks, consistency

CFG = load_config()
SPEC = checks.load_rules()["consistency"]


def monthly_series(start, end, values):
    dates = pd.date_range(f"{start}-01-01", f"{end}-12-01", freq="MS")
    return pd.DataFrame({"date": dates, "anomaly": values(dates)})


# ------------------------------------------------------------------ series helpers
def test_rebasing_removes_the_seasonal_cycle():
    frame = monthly_series(1990, 2021, lambda d: 10 + 5 * np.sin(2 * np.pi * d.month / 12))
    rebased = consistency.rebase_monthly(frame, "anomaly", (1991, 2020))
    assert np.allclose(rebased, 0.0)


def test_rebasing_requires_the_whole_reference_period():
    frame = monthly_series(1995, 2020, lambda d: np.zeros(len(d)))
    with pytest.raises(ValueError, match="rebasing needs 30"):
        consistency.rebase_monthly(frame, "anomaly", (1991, 2020))


def test_annual_means_need_twelve_months():
    dates = pd.Series(pd.date_range("2000-01-01", "2001-11-01", freq="MS"))
    out = consistency.annual_means(dates, pd.Series(np.ones(len(dates))))
    assert out.index.tolist() == [2000]


def test_newey_west_rule():
    assert consistency.newey_west_lags(60) == 3
    assert consistency.newey_west_lags(145) == 4


# ------------------------------------------------------------------ global records
def test_global_comparison_finds_early_divergence():
    rng = np.random.default_rng(0)
    dates = pd.date_range("1880-01-01", "2024-12-01", freq="MS")
    signal = rng.normal(0, 0.15, len(dates)).cumsum() * 0.02 + rng.normal(0, 0.1, len(dates))

    def source(early_bias):
        noise = rng.normal(0, 0.03, len(dates))
        bias = np.where(dates.year < 1920, early_bias, 0.0)
        return pd.DataFrame({"date": dates, "anomaly": signal + noise + bias})

    series = {"a": source(0.0), "b": source(0.0), "c": source(0.2)}
    out = consistency.global_comparison(
        series, (1880, 2024), SPEC["periods"], (1991, 2020), SPEC["global_diff_corr"]
    )
    out.pop("annual")
    early, late = "1880-1919", "1980-2024"
    ac = out["pairs"]["a-c"]["annual_difference_by_period"]
    assert ac[early]["rms"] > 0.15 > ac[late]["rms"]
    assert out["pairs"]["a-c"]["largest_abs_difference"]["year"] < 1920
    assert out["pairs"]["a-b"]["annual_corr_changes"] > 0.9
    assert out["pairs"]["a-b"]["score"] == "Good"


# ------------------------------------------------------------------ Natural Earth codes
def test_iso3_codes_handle_the_minus_99_traps():
    frame = pd.DataFrame(
        {
            "NAME": ["France", "Australia", "Indian Ocean Ter.", "Kosovo", "Kenya"],
            "ISO_A3": ["-99", "AUS", "-99", "-99", "KEN"],
            "ISO_A3_EH": ["FRA", "AUS", "AUS", "-99", "KEN"],
            "ADM0_A3": ["FRA", "AUS", "IOA", "KOS", "KEN"],
        }
    )
    assert consistency.iso3_codes(frame).tolist() == ["FRA", "AUS", "IOA", "XKX", "KEN"]


def test_iso3_codes_reject_duplicates():
    frame = pd.DataFrame(
        {
            "NAME": ["A", "B"],
            "ISO_A3": ["KEN", "KEN"],
            "ISO_A3_EH": ["KEN", "KEN"],
            "ADM0_A3": ["KEN", "KEN"],
        }
    )
    with pytest.raises(ValueError, match="not unique"):
        consistency.iso3_codes(frame)


def toy_countries():
    """Two boxes: AAA straddles 0 deg longitude (tests 0-360 wrapping), BBB in the east."""
    return gpd.GeoDataFrame(
        {"NAME": ["Alpha", "Beta"], "iso3": ["AAA", "BBB"]},
        geometry=[box(-10.1, -5.1, 10.1, 5.1), box(99.9, 19.9, 110.1, 30.1)],
        crs="EPSG:4326",
    )


def test_country_cells_wrap_longitudes_and_count_cells():
    lat = np.arange(90.0, -91.0, -5.0)
    lon = np.arange(0.0, 360.0, 5.0)
    cells = consistency.country_cells(toy_countries(), ["AAA", "BBB"], lat, lon)
    assert cells["AAA"]["n_cells"] == 5 * 3  # lon 350, 355, 0, 5, 10 x lat -5, 0, 5
    assert cells["BBB"]["n_cells"] == 3 * 3
    with pytest.raises(ValueError, match="not in Natural Earth"):
        consistency.country_cells(toy_countries(), ["ZZZ"], lat, lon)


# ------------------------------------------------------------------ ERA5 vs Berkeley
def test_step_test_detects_a_satellite_era_shift():
    rng = np.random.default_rng(1)
    years = pd.Index(range(1961, 2021))
    shifted = pd.Series(np.where(years >= 1979, 0.5, 0.0) + rng.normal(0, 0.05, 60), index=years)
    flat = pd.Series(rng.normal(0, 0.05, 60), index=years)
    out = consistency.step_test(shifted, 1979)
    assert out["step_degc"] == pytest.approx(0.5, abs=0.05)
    assert out["p"] < 0.001
    assert out["hac_lags"] == 3
    assert consistency.step_test(flat, 1979)["p"] > 0.05


def anomalies(code, values, years=range(1961, 2021)):
    return pd.DataFrame({"iso3": code, "year": list(years), "anom": values})


def test_country_comparison_trips_only_on_real_steps():
    rng = np.random.default_rng(2)
    years = np.arange(1961, 2021)
    truth = rng.normal(0, 0.4, len(years))
    bk = pd.concat([anomalies("AAA", truth), anomalies("BBB", truth)])
    era5 = pd.concat(
        [
            anomalies("AAA", truth + rng.normal(0, 0.05, len(years))),
            anomalies("BBB", truth + np.where(years >= 1979, 0.4, 0.0) + rng.normal(0, 0.05, 60)),
        ]
    )
    out = consistency.country_comparison(
        era5, bk, (1961, 2020), 1979, 0.2, 0.05, SPEC["country_diff_corr"]
    )
    assert out["AAA"]["rd5"]["tripwire"] is False
    assert out["BBB"]["rd5"]["tripwire"] is True
    assert out["AAA"]["corr_changes"] > 0.95
    assert out["BBB"]["rd5"]["p_holm"] >= out["BBB"]["rd5"]["p"]


def test_annual_anomalies_skip_series_without_full_reference_period():
    dates = pd.date_range("1961-01-01", "2016-12-01", freq="MS")
    frame = pd.DataFrame({"iso3": "OLD", "date": dates, "anomaly": 1.0})
    assert consistency.annual_anomalies(frame, "anomaly", (1991, 2020)).empty


def era5_zip(tmp_path):
    lat = np.arange(90.0, -91.0, -5.0)
    lon = np.arange(0.0, 360.0, 5.0)
    times = pd.date_range("2000-01-01", periods=24, freq="MS")
    t2m = np.full((24, lat.size, lon.size), 273.15)
    t2m[:, (lat >= 20) & (lat <= 30), :] += 10.0  # BBB box 10 degC warmer
    tp = np.zeros_like(t2m)
    for name, var, data in (("a.nc", "t2m", t2m), ("b.nc", "tp", tp)):
        xr.Dataset(
            {var: (("valid_time", "latitude", "longitude"), data)},
            coords={"valid_time": times, "latitude": lat, "longitude": lon},
        ).to_netcdf(tmp_path / name)
    (tmp_path / "era5").mkdir()
    with zipfile.ZipFile(tmp_path / "era5" / "x.zip", "w") as archive:
        archive.write(tmp_path / "a.nc", "a.nc")
        archive.write(tmp_path / "b.nc", "b.nc")
    return ManifestEntry(
        "era5_monthly_single_levels", "era5/x.zip", "0" * 64, 0, "", "", "", "", "cdsapi"
    )


def test_era5_country_monthly_area_weighted(tmp_path):
    entry = era5_zip(tmp_path)
    frame, cells = consistency.era5_country_monthly(
        [entry], tmp_path, ["AAA", "BBB"], toy_countries()
    )
    means = frame.groupby("iso3")["t2m_c"].mean()
    assert means["AAA"] == pytest.approx(0.0, abs=1e-6)
    assert means["BBB"] == pytest.approx(10.0, abs=1e-6)
    assert len(frame) == 24 * 2
    assert cells["AAA"]["n_cells"] == 15


def test_figures_render(tmp_path):
    annual = pd.DataFrame({"a": [0.0, 0.1], "b": [0.05, 0.1], "c": [0.0, 0.0]}, index=[2000, 2001])
    consistency.global_difference_figure(annual, tmp_path / "g.png")
    results = {k: {"rd5": {"step_degc": s, "se": 0.05}} for k, s in (("AAA", 0.0), ("BBB", 0.4))}
    consistency.rd5_figure(results, 0.2, 1979, tmp_path / "r.png")
    assert (tmp_path / "g.png").read_bytes()[:4] == b"\x89PNG"
    assert (tmp_path / "r.png").read_bytes()[:4] == b"\x89PNG"


def test_run_end_to_end(monkeypatch, tmp_path):
    entry = era5_zip(tmp_path)
    glob = monthly_series(1880, 2024, lambda d: np.sin(np.arange(len(d)) / 7.0))
    dates = pd.date_range("1961-01-01", "2020-12-01", freq="MS")
    bk = pd.concat(
        [
            pd.DataFrame({"iso3": c, "date": dates, "anomaly": np.cos(np.arange(len(dates)))})
            for c in ("AAA", "BBB")
        ]
    )
    headers = pd.DataFrame({"iso3": ["AAA", "BBB"], "area_km2": [1.0e6, 2.0e5]})

    def t(name, frame):
        return RawTable("d", name, frame, ["f"])

    tables = {
        "gistemp_v4_global_monthly": [t("monthly", glob)],
        "hadcrut5_global_monthly": [t("monthly", glob)],
        "berkeley_global_monthly": [t("air_over_sea_ice", glob)],
        "berkeley_country_monthly": [t("monthly", bk), t("file_headers", headers)],
    }
    rules = checks.load_rules()
    rules["consistency"]["countries"] = ["AAA", "BBB"]
    monkeypatch.setattr(consistency, "read_dataset", lambda name, manifest, root: tables[name])
    monkeypatch.setattr(consistency, "natural_earth_countries", toy_countries)
    monkeypatch.setattr(consistency, "load_rules", lambda: rules)
    out = consistency.run([entry], tmp_path, CFG, tmp_path / "figures")
    assert set(out["global"]["pairs"]) == {
        "gistemp-hadcrut5",
        "gistemp-berkeley",
        "hadcrut5-berkeley",
    }
    assert out["global"]["pairs"]["gistemp-hadcrut5"]["annual_corr_levels"] == pytest.approx(1.0)
    # the toy ERA5 file has only 2000-2001, so no country has a full 1991-2020 baseline
    assert out["countries"] == {}
    assert (tmp_path / "figures" / "consistency_global_differences.png").exists()
