"""Raw-reader tests on small synthetic files in each provider's layout (offline)."""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from climate_impact.ingest import readers
from climate_impact.provenance import ManifestEntry


def entry(dataset, file, **request):
    return ManifestEntry(
        dataset=dataset,
        file=file,
        sha256="0" * 64,
        size_bytes=0,
        source_url="",
        version="",
        access_date="2026-11-18",
        license="",
        retrieval="http",
        request=request,
    )


def write(root, rel, data):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    return rel


def make_zip(members):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


# ------------------------------------------------------------------ strict parsing
def test_strict_parse_counts_blanks_and_sentinels():
    values = pd.Series(["1.5", "", "***", " -2 ", "-99.99"], name="x")
    parsed, counts = readers.to_numeric_strict(
        values, text_sentinels=["***"], numeric_sentinels=[-99.99]
    )
    assert parsed.isna().tolist() == [False, True, True, False, True]
    assert parsed.iloc[3] == -2.0
    assert counts == {"blank": 1, "***": 1, "-99.99": 1}


def test_strict_parse_rejects_unknown_tokens():
    with pytest.raises(ValueError, match="unexpected non-numeric"):
        readers.to_numeric_strict(pd.Series(["1", "n/a"], name="x"))


def test_numeric_sentinel_does_not_touch_close_real_values():
    parsed, counts = readers.to_numeric_strict(
        pd.Series(["-1", "-1.01", "1"]), numeric_sentinels=[-1]
    )
    assert parsed.isna().sum() == 1
    assert counts == {"-1": 1}


# ------------------------------------------------------------------ global series
GISTEMP = (
    "Land-Ocean: Global Means\n"
    "Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec,J-D,D-N,DJF,MAM,JJA,SON\n"
    "1880,-.19,-.25,-.10,-.17,-.10,-.21,-.19,-.11,-.15,-.23,-.22,-.18,-.18,***,***,-.12,-.17,-.20\n"
    "1881,-.20,-.14,.03,.05,.06,-.19,.00,-.04,-.15,-.22,***,***,***,***,-.09,.05,-.08,-.17\n"
)


def test_gistemp_long_with_sentinels(tmp_path):
    rel = write(tmp_path, "gistemp/GLB.csv", GISTEMP)
    (table,) = readers.read_gistemp([entry("gistemp_v4_global_monthly", rel)], tmp_path)
    frame = table.frame
    assert len(frame) == 24
    assert frame["date"].is_monotonic_increasing
    assert frame["date"].iloc[0] == pd.Timestamp("1880-01-01")
    assert frame.loc[frame["date"] == "1880-03-01", "anomaly"].item() == pytest.approx(-0.10)
    assert table.sentinels == {"value: ***": 2}  # only Nov/Dec 1881; J-D etc. not read
    assert frame["anomaly"].isna().sum() == 2


def test_hadcrut5(tmp_path):
    text = (
        "Time,Anomaly (deg C),Lower confidence limit (2.5%),Upper confidence limit (97.5%)\n"
        "1850-01,-0.67,-0.98,-0.36\n1850-02,-0.33,-0.70,0.03\n"
    )
    rel = write(tmp_path, "hadcrut5/global_monthly.csv", text)
    (table,) = readers.read_hadcrut5([entry("hadcrut5_global_monthly", rel)], tmp_path)
    assert list(table.frame.columns) == ["date", "anomaly", "lower_2p5", "upper_97p5"]
    assert table.frame["date"].iloc[1] == pd.Timestamp("1850-02-01")


def test_hadcrut5_layout_change_is_detected(tmp_path):
    rel = write(tmp_path, "hadcrut5/global_monthly.csv", "Time,Anomaly\n1850-01,-0.6\n")
    with pytest.raises(ValueError, match="layout changed"):
        readers.read_hadcrut5([entry("hadcrut5_global_monthly", rel)], tmp_path)


ROW = "  {y}  {m:>4}  {a:>8}  {u}" + "       NaN    NaN" * 4 + "\n"


def berkeley_text(sections):
    lines = ["% Rohde, R. A. and Hausfather, Z.: ESSD, 12, 3469\x963479\n", "% \n"]
    for title, rows in sections:
        if title:
            lines.append(f"% Global Average Temperature Anomaly with Sea Ice {title}\n")
        lines.append("% Year, Month,  Anomaly, Unc., ...\n \n")
        lines += [ROW.format(y=y, m=m, a=a, u="0.365") for y, m, a in rows]
    return "".join(lines).encode("latin-1")


def test_berkeley_global_two_tables(tmp_path):
    data = berkeley_text(
        [
            ("Temperature Inferred from Air Temperatures", [(1850, 1, "-0.753"), (1850, 2, "NaN")]),
            ("Temperature Inferred from Water Temperatures", [(1850, 1, "-0.717")]),
        ]
    )
    rel = write(tmp_path, "berkeley/Land_and_Ocean_complete.txt", data)
    tables = readers.read_berkeley_global([entry("berkeley_global_monthly", rel)], tmp_path)
    by_name = {t.table: t for t in tables}
    assert set(by_name) == {"air_over_sea_ice", "water_under_sea_ice"}
    assert len(by_name["air_over_sea_ice"].frame) == 2
    assert by_name["air_over_sea_ice"].sentinels == {"anomaly: NaN": 1}
    assert by_name["water_under_sea_ice"].frame["anomaly"].item() == pytest.approx(-0.717)


def test_berkeley_global_missing_section_raises(tmp_path):
    data = berkeley_text([("Temperature Inferred from Air Temperatures", [(1850, 1, "-0.7")])])
    rel = write(tmp_path, "berkeley/Land_and_Ocean_complete.txt", data)
    with pytest.raises(ValueError, match="sections"):
        readers.read_berkeley_global([entry("berkeley_global_monthly", rel)], tmp_path)


COUNTRY_HEADER = (
    "% method for the region: \n% \n%    France\n% \n"
    "% This analysis was run on 06-Jan-2021 15:35:34\n"
    "%%   Latitude Range: -49.72 to 51.09\n%%   Longitude Range: -178.19 to 171.34\n"
    "%%   Area: 658275.47 km^2\n%%   Approximate number of temperature stations: 329\n"
    "%%  Estimated Jan 1951-Dec 1980 absolute temperature (C): 12.96 +/- 0.14\n"
)


def test_berkeley_country_uses_manifest_iso3_and_header(tmp_path):
    body = "".join(ROW.format(y=2020, m=m, a="1.1", u="0.2") for m in (11, 12))
    rel = write(tmp_path, "berkeley/country/france-TAVG-Trend.txt", COUNTRY_HEADER + body)
    data, header = readers.read_berkeley_country(
        [entry("berkeley_country_monthly", rel, iso3="FRA")], tmp_path
    )
    assert data.frame["iso3"].unique().tolist() == ["FRA"]
    row = header.frame.iloc[0]
    assert row["region"] == "France"
    assert row["n_stations"] == 329
    assert row["lat_range"] == "-49.72 to 51.09"
    assert row["last_valid"] == pd.Timestamp("2020-12-01")


def test_berkeley_wrong_field_count_raises(tmp_path):
    rel = write(tmp_path, "berkeley/country/x.txt", "  2020  1  0.5  0.1\n")
    with pytest.raises(ValueError, match="expected 12"):
        readers.read_berkeley_country([entry("berkeley_country_monthly", rel, iso3="X")], tmp_path)


def test_noaa_co2_column_specific_codes(tmp_path):
    text = (
        "# comment line\n# another\n"
        "year,month,decimal date,average,deseasonalized,ndays,sdev,unc\n"
        "1958,3,1958.2027,315.71,314.44,-1,-9.99,-0.99\n"
        "2025,1,2025.0417,-99.99,-99.99,25,0.5,0.1\n"
    )
    rel = write(tmp_path, "noaa/co2_mm_mlo.csv", text)
    (table,) = readers.read_noaa_co2([entry("noaa_co2_mlo_monthly", rel)], tmp_path)
    frame = table.frame
    assert frame["average"].iloc[0] == pytest.approx(315.71)
    assert frame["ndays"].isna().tolist() == [True, False]
    assert table.sentinels == {
        "average: -99.99": 1,
        "deseasonalized: -99.99": 1,
        "ndays: -1": 1,
        "sdev: -9.99": 1,
        "unc: -0.99": 1,
    }


def test_oni_centre_month(tmp_path):
    text = " SEAS  YR   TOTAL   ANOM\n  DJF 1950  24.72  -1.53\n  NDJ 1950  25.01  -0.80\n"
    rel = write(tmp_path, "noaa/oni.ascii.txt", text)
    (table,) = readers.read_oni([entry("noaa_oni", rel)], tmp_path)
    assert table.frame["date"].tolist() == [pd.Timestamp("1950-01-01"), pd.Timestamp("1950-12-01")]


def test_oni_unknown_season_raises(tmp_path):
    rel = write(tmp_path, "noaa/oni.ascii.txt", " SEAS  YR   TOTAL   ANOM\n  XYZ 1950  1  2\n")
    with pytest.raises(ValueError, match="unknown season"):
        readers.read_oni([entry("noaa_oni", rel)], tmp_path)


# ------------------------------------------------------------------ country panels
FAO_HEADER = (
    "Area Code,Area Code (M49),Area,Item Code,Item Code (CPC),Item,Element Code,Element,"
    "Year Code,Year,Unit,Value,Flag,Note\n"
)


def test_faostat_latin1_fallback_and_types(tmp_path):
    rows = (
        "107,'384,C\xf4te d'Ivoire,56,'0112,Maize (corn),5412,Yield,2020,2020,kg/ha,2100.5,A,\n"
        "107,'384,C\xf4te d'Ivoire,56,'0112,Maize (corn),5312,Area harvested,2020,2020,ha,,M,\n"
    )
    main = (FAO_HEADER + rows).encode("latin-1")
    flags = b"Flag,Flags\nA,Official figure\nM,Missing value\n"
    archive = make_zip(
        {
            "Production_Crops_Livestock_E_All_Data_(Normalized).csv": main,
            "Production_Crops_Livestock_E_Flags.csv": flags,
        }
    )
    rel = write(tmp_path, "faostat/fao.zip", archive)
    production, legend = readers.read_faostat([entry("faostat_crops_livestock", rel)], tmp_path)
    frame = production.frame
    assert "Year Code" not in frame
    assert frame["Area"].iloc[0] == "C\xf4te d'Ivoire"
    assert frame["Value"].isna().tolist() == [False, True]
    assert production.sentinels == {"Value: blank": 1}
    assert any("latin-1" in n for n in production.notes)
    assert len(legend.frame) == 2


def test_faostat_utf8_is_preferred(tmp_path):
    row = "107,'384,C\xf4te d'Ivoire,56,'0112,Maize,5412,Yield,2020,2020,kg/ha,1,A,\n"
    archive = make_zip({"X_All_Data_(Normalized).csv": (FAO_HEADER + row).encode("utf-8")})
    rel = write(tmp_path, "faostat/fao.zip", archive)
    (production,) = readers.read_faostat([entry("faostat_crops_livestock", rel)], tmp_path)
    assert production.frame["Area"].iloc[0] == "C\xf4te d'Ivoire"
    assert any("decoded as utf-8" in n for n in production.notes)


WDI_INDICATORS = ["SP.POP.TOTL", "NY.GDP.PCAP.PP.KD", "NV.AGR.TOTL.ZS", "SP.URB.TOTL.IN.ZS"]


def wdi_zip(indicators):
    lines = ["Country Name,Country Code,Indicator Name,Indicator Code,1960,1961,"]
    for code in indicators:
        lines.append(f"Kenya,KEN,{code} name,{code},1.5,")
    lines.append("Kenya,KEN,Other,XX.OTHER,9,9,")
    country = (
        "Country Code,Short Name,Region,Income Group\n"
        "KEN,Kenya,Sub-Saharan Africa,Lower middle income\nWLD,World,,\n"
    )
    return make_zip({"WDICSV.csv": "\n".join(lines) + "\n", "WDICountry.csv": country})


def test_wdi_filters_indicators_and_melts(tmp_path):
    rel = write(tmp_path, "wdi/WDI_CSV.zip", wdi_zip([*WDI_INDICATORS, "AG.LND.TOTL.K2"]))
    data, country = readers.read_wdi([entry("wdi", rel)], tmp_path)
    assert set(data.frame["Indicator Code"]) == {*WDI_INDICATORS, "AG.LND.TOTL.K2"}
    assert sorted(data.frame["year"].unique()) == [1960, 1961]
    assert data.frame["value"].isna().sum() == 5  # every 1961 cell is blank
    assert country.frame.loc[1, "Region"] == ""


def test_wdi_missing_indicator_raises(tmp_path):
    rel = write(tmp_path, "wdi/WDI_CSV.zip", wdi_zip(WDI_INDICATORS))
    with pytest.raises(ValueError, match=r"AG\.LND\.TOTL\.K2"):
        readers.read_wdi([entry("wdi", rel)], tmp_path)


def test_emdat_excel(tmp_path):
    path = tmp_path / "emdat" / "emdat_export.xlsx"
    path.parent.mkdir(parents=True)
    events = pd.DataFrame(
        {
            "DisNo.": ["2000-0001-KEN", "2000-0002-KEN"],
            "ISO": ["KEN", "KEN"],
            "Disaster Subgroup": ["Hydrological", "Geophysical"],
            "Start Year": [2000, 2000],
            "Total Deaths": [12, np.nan],
            "Total Affected": [1000, 50],
        }
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        events.to_excel(writer, sheet_name="EM-DAT Data", index=False)
        pd.DataFrame({"info": ["version"]}).to_excel(writer, sheet_name="EM-DAT Info", index=False)
    (table,) = readers.read_emdat([entry("emdat", "emdat/emdat_export.xlsx")], tmp_path)
    assert table.frame.shape == (2, 6)
    assert "EM-DAT Info" in table.notes[0]


def test_nd_gain_skips_macos_artefacts(tmp_path):
    csv = "ISO3,Name,1995,1996\nKEN,Kenya,0.52,\nNOR,Norway,0.29,0.28\n"
    archive = make_zip(
        {
            "release/vulnerability/vulnerability.csv": csv,
            "__MACOSX/release/vulnerability/._vulnerability.csv": b"\x00\x05junk",
            "release/readiness/readiness.csv": csv,
        }
    )
    rel = write(tmp_path, "nd_gain/nd_gain.zip", archive)
    vulnerability, readiness = readers.read_nd_gain([entry("nd_gain", rel)], tmp_path)
    assert vulnerability.table == "vulnerability"
    assert len(vulnerability.frame) == 4
    assert vulnerability.sentinels == {"value: blank": 1}
    assert readiness.frame["year"].tolist() == [1995, 1995, 1996, 1996]


def test_read_dataset_requires_registered_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="no files registered"):
        readers.read_dataset("noaa_oni", [], tmp_path)


def test_every_core_tabular_dataset_has_a_reader():
    from climate_impact.config import load_datasets

    catalogue = load_datasets()
    tabular = {
        n
        for n, m in catalogue.items()
        if m["priority"] == "core"
        and m["access"] != "package"
        and n != "era5_monthly_single_levels"
    }
    assert tabular == set(readers.READERS)
