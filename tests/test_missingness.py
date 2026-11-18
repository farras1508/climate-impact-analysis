"""Step 3.4 missingness tests: simulated worlds with known missingness mechanisms."""

import numpy as np
import pandas as pd
import pytest

from climate_impact.config import load_config
from climate_impact.ingest.readers import RawTable
from climate_impact.quality import checks, missingness

CFG = load_config()
INCOME = missingness.INCOME_ORDER
TERMS = ["C(income, Treatment('High income'))", "C(region)", "log_pop", "decade"]


def world(seed, probs):
    """160 countries x 1990-2020; ``probs`` maps income group -> P(missing)."""
    rng = np.random.default_rng(seed)
    countries = pd.DataFrame(
        {
            "iso3": [f"C{i:03d}" for i in range(160)],
            "income": [INCOME[i % 4] for i in range(160)],
            "region": [f"R{(i // 4) % 4}" for i in range(160)],
        }
    )
    grid = missingness.country_year_grid(countries, (1990, 2020), {})
    grid["log_pop"] = rng.normal(size=len(grid))
    grid["decade"] = (grid["year"] - 2000) / 10
    p = grid["income"].map(probs).to_numpy()
    grid["missing"] = (rng.random(len(grid)) < p).astype(int)
    return grid


def test_income_driven_missingness_is_detected():
    grid = world(42, {INCOME[0]: 0.4, INCOME[1]: 0.25, INCOME[2]: 0.1, INCOME[3]: 0.03})
    out = missingness.fit_missingness(grid, TERMS, 15.0, 0.05)
    assert out["significant_terms"] == ["income group"]
    assert out["classification"].startswith("not MCAR")
    assert "MNAR cannot be tested" in out["classification"]
    assert out["odds_ratios"]["C(income, Treatment('High income'))[T.Low income]"]["or"] > 5


def test_random_missingness_is_not_flagged():
    grid = world(7, dict.fromkeys(INCOME, 0.15))
    out = missingness.fit_missingness(grid, TERMS, 15.0, 0.05)
    assert out["term_pvalues"]["year"] < 0.05  # a chance "finding" without correction ...
    assert out["term_pvalues_holm"]["year"] > 0.05  # ... removed by Holm
    assert out["significant_terms"] == []
    assert out["classification"].startswith("MCAR not rejected")
    assert not out["separation"]


def test_false_alarm_rate_is_controlled_across_random_worlds():
    flagged = sum(
        bool(
            missingness.fit_missingness(
                world(seed, dict.fromkeys(INCOME, 0.15)), TERMS, 15.0, 0.05
            )["significant_terms"]
        )
        for seed in range(100, 130)
    )
    # Holm keeps the family-wise error <= 5 %: expect ~1.5 of 30; P(>= 6 | p = 0.05) < 1 %
    assert flagged <= 5


def test_no_variation_is_reported_not_fitted():
    grid = world(1, dict.fromkeys(INCOME, 0.0))
    out = missingness.fit_missingness(grid, TERMS, 15.0, 0.05)
    assert out == {
        "n": len(grid),
        "n_countries": 160,
        "pct_missing": 0.0,
        "n_model": len(grid),
        "n_dropped_no_population": 0,
        "estimable": False,
        "reason": "no variation in missingness",
    }


def test_rows_without_population_are_counted_not_hidden():
    grid = world(4, dict.fromkeys(INCOME, 0.2))
    grid.loc[grid["iso3"] == "C000", "log_pop"] = np.nan
    out = missingness.fit_missingness(grid, TERMS, 15.0, 0.05)
    assert out["n"] == len(grid)
    assert out["n_dropped_no_population"] == 31  # C000 x 1990-2020


def test_constant_population_does_not_erase_rows():
    grid = pd.DataFrame({"iso3": ["A", "B"], "year": [2024, 2024]})
    pop = pd.DataFrame({"Country Code": ["A", "B"], "year": 2024, "value": [1e6, 1e6]})
    assert missingness.add_log_pop(grid, pop)["log_pop"].tolist() == [0.0, 0.0]


def test_separation_is_flagged():
    grid = world(3, {INCOME[0]: 0.5, INCOME[1]: 0.0, INCOME[2]: 0.0, INCOME[3]: 0.0})
    out = missingness.fit_missingness(grid, TERMS, 15.0, 0.05)
    assert out["separation"]


def test_constant_terms_are_dropped_from_the_formula():
    grid = world(5, dict.fromkeys(INCOME, 0.2)).assign(region="only one")
    out = missingness.fit_missingness(grid, TERMS, 15.0, 0.05)
    assert "C(region)" not in out["formula"]


def test_universe_and_entry_years():
    classes = pd.DataFrame(
        {
            "Country Code": ["KEN", "WLD", "VEN", "SSD"],
            "Region": ["Sub-Saharan Africa", "", "Latin America & Caribbean", "Sub-Saharan Africa"],
            "Income Group": ["Lower middle income", "", "", "Low income"],
        }
    )
    uni = missingness.universe(classes)
    assert uni["iso3"].tolist() == ["KEN", "SSD", "VEN"]
    assert uni.set_index("iso3").loc["VEN", "income"] == "Unclassified"
    entry = missingness.entry_years(checks.load_rules()["domain"]["successions"])
    assert entry["SSD"] == 2012
    assert entry["RUS"] == 1992
    assert entry["SRB"] == 2006  # latest of Yugoslavia (1992) and Serbia-Montenegro (2006)
    grid = missingness.country_year_grid(uni, (2010, 2013), entry)
    assert grid.groupby("iso3")["year"].min().to_dict() == {"KEN": 2010, "SSD": 2012, "VEN": 2010}


def test_log_pop_fills_missing_years_with_country_median():
    grid = pd.DataFrame({"iso3": ["A", "A", "B"], "year": [2000, 2001, 2000]})
    pop = pd.DataFrame({"Country Code": ["A", "B"], "year": [2000, 2000], "value": [1e6, 1e8]})
    out = missingness.add_log_pop(grid, pop)
    assert out["log_pop"].notna().all()
    assert out.loc[0, "log_pop"] == out.loc[1, "log_pop"]


def test_describe_and_heatmap(tmp_path):
    grid = world(9, {INCOME[0]: 0.5, INCOME[1]: 0.2, INCOME[2]: 0.1, INCOME[3]: 0.0})
    out = missingness.describe(grid)
    assert out["by_income"]["High income"] == 0.0
    assert set(out["by_decade"]) == {"1990s", "2000s", "2010s", "2020s"}
    path = tmp_path / "fig" / "m.png"
    missingness.heatmap(grid, path, "test")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_h3_grid_classifies_leading_internal_trailing():
    fao = pd.DataFrame(
        {
            "Area Code": "114",
            "Area Code (M49)": "'404",
            "Area": "Kenya",
            "Item Code": "56",
            "Element Code": "5412",
            "Year": [2001, 2003, 2004],
            "Value": [1.0, 1.0, 1.0],
        }
    ).astype({"Area Code": "category", "Element Code": "category"})
    countries = pd.DataFrame({"iso3": ["KEN"], "income": ["Lower middle income"], "region": ["R"]})
    grid, kinds = missingness.h3_yield_grid(fao, countries, (2000, 2005), {}, 5412)
    assert kinds == {"observed": 3, "leading": 1, "internal": 1, "trailing": 1}
    assert grid["missing"].sum() == 3


def test_run_end_to_end(monkeypatch, tmp_path):
    codes = ["KEN", "NOR", "SSD"]
    wdi = pd.DataFrame(
        [
            (c, ind, y, np.nan if (c == "SSD" and ind != "SP.POP.TOTL") else 1e6)
            for c in codes
            for ind in ("SP.POP.TOTL", "NY.GDP.PCAP.PP.KD", "NV.AGR.TOTL.ZS")
            for y in range(1989, 2025)
        ],
        columns=["Country Code", "Indicator Code", "year", "value"],
    )
    classes = pd.DataFrame(
        {
            "Country Code": codes,
            "Region": ["Sub-Saharan Africa", "Europe & Central Asia", "Sub-Saharan Africa"],
            "Income Group": ["Lower middle income", "High income", "Low income"],
        }
    )
    fao = pd.DataFrame(
        {
            "Area Code": ["114", "162"],
            "Area Code (M49)": ["'404", "'578"],
            "Area": ["Kenya", "Norway"],
            "Item Code": ["56", "15"],
            "Element Code": ["5412", "5412"],
            "Year": [2000, 2000],
            "Value": [1.0, 1.0],
        }
    ).astype(
        dict.fromkeys(
            ["Area Code", "Area Code (M49)", "Area", "Item Code", "Element Code"], "category"
        )
    )
    nd = pd.DataFrame({"ISO3": ["KEN", "NOR"], "Name": "x", "year": 2024, "value": 0.5})

    def t(name, frame):
        return RawTable("d", name, frame, ["f"])

    tables = {
        "wdi": [t("indicators", wdi), t("country_classification", classes)],
        "faostat_crops_livestock": [t("production", fao)],
        "nd_gain": [t("vulnerability", nd), t("readiness", nd)],
    }
    monkeypatch.setattr(missingness, "read_dataset", lambda name, manifest, root: tables[name])
    out = missingness.run([], tmp_path, CFG, tmp_path / "figures")
    variables = out["variables"]
    assert set(variables) == set(checks.load_rules()["missingness"]["variables"])
    assert variables["h4_gdp_pc_ppp"]["describe"]["countries_most_missing_pct"] == {"SSD": 100.0}
    vulnerability = variables["h5_vulnerability"]["model"]
    assert vulnerability["pct_missing"] == pytest.approx(100 / 3, abs=0.01)
    # with 3 countries some models cannot be identified: reported, never a crash
    assert all("estimable" in v["model"] for v in variables.values())
    assert (tmp_path / "figures" / "missingness_h3_yield.png").exists()
    page = missingness.render(out)
    assert "| `h3_yield` |" in page
