"""Raw-file readers: one function per dataset, returning tables *as read*.

Rules (``docs/data_dictionary.md``, section 3):

* Sentinel codes become NaN **when the file is read**, and every replacement is counted.
* Text that is neither a number, blank, nor a declared sentinel raises ``ValueError``:
  an unexpected token is evidence of a format change and must never become NaN silently.
* No unit conversion, rebasing, filtering, deduplication or aggregation: those are
  Phase 4 decisions, taken only after the Phase 3 quality assessment.
* Reshaping is limited to wide -> long where the provider delivers a wide layout
  (GISTEMP, WDI, ND-GAIN), so every table has one value per row.

Readers locate files through the manifest, so they read exactly the registered vintage.
"""

from __future__ import annotations

import logging
import re
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from climate_impact.config import PROJECT_ROOT, load_datasets
from climate_impact.provenance import ManifestEntry

logger = logging.getLogger(__name__)

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
ONI_SEASONS = ("DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ")
BERKELEY_COLUMNS = 12  # year, month, then (anomaly, unc.) for 1-, 12-, 60-, 120-, 240-month means
YEAR_COLUMN = re.compile(r"^\d{4}$")


@dataclass
class RawTable:
    """One table read from raw files, with the facts needed to profile it.

    Attributes
    ----------
    dataset, table : str
        Manifest dataset name and table name within it.
    frame : pandas.DataFrame
        Data as read (sentinels already NaN).
    source_files : list of str
        Manifest-relative paths of the files read.
    time_col, entity_col : str or None
        Column identifying the period / the country (``None`` if not applicable).
    time_freq : str or None
        ``"M"`` (monthly) or ``"Y"`` (annual).
    value_cols : list of str
        Columns holding the measured quantities.
    key : list of str
        Columns that should identify a row uniquely (checked in Step 3.2).
    sentinels : dict of str to int
        ``"column: code"`` -> number of values replaced by NaN at read.
    notes : list of str
        Facts about the read (encoding, members, file timestamps, dropped fields).
    """

    dataset: str
    table: str
    frame: pd.DataFrame
    source_files: list[str]
    time_col: str | None = None
    entity_col: str | None = None
    time_freq: str | None = None
    value_cols: list[str] = field(default_factory=list)
    key: list[str] = field(default_factory=list)
    sentinels: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- strict parsing helpers
def to_numeric_strict(
    values: pd.Series,
    *,
    text_sentinels: Sequence[str] = (),
    numeric_sentinels: Sequence[float] = (),
) -> tuple[pd.Series, dict[str, int]]:
    """Parse text to float64; blanks and declared sentinels -> NaN, anything else raises.

    Parameters
    ----------
    values : pandas.Series
        Raw values (text or numbers).
    text_sentinels : sequence of str
        Exact tokens meaning "missing", e.g. ``"***"`` (GISTEMP) or ``"NaN"`` (Berkeley).
    numeric_sentinels : sequence of float
        Numbers meaning "missing", e.g. ``-99.99`` (NOAA), compared with ``np.isclose``.

    Returns
    -------
    (pandas.Series, dict)
        Parsed values and the count of each code replaced (``"blank"`` for empty cells).

    Raises
    ------
    ValueError
        If any value is neither numeric, blank, nor a declared sentinel.
    """
    text = values.astype("string").fillna("").str.strip()
    missing = text.eq("").to_numpy(dtype=bool)
    counts = {"blank": int(missing.sum())}
    for code in text_sentinels:
        hit = text.eq(code).to_numpy(dtype=bool)
        counts[code] = int(hit.sum())
        missing |= hit
    parsed = pd.to_numeric(text.mask(missing), errors="coerce").astype("float64")
    bad = parsed.isna().to_numpy() & ~missing
    if bad.any():
        examples = sorted(set(text[bad].tolist()))[:5]
        raise ValueError(
            f"{values.name}: {int(bad.sum())} unexpected non-numeric value(s) {examples}"
        )
    for number in numeric_sentinels:
        hit = np.isclose(parsed.to_numpy(), number, rtol=0.0, atol=1e-9)
        counts[f"{number:g}"] = int(hit.sum())
        parsed = parsed.mask(hit)
    return parsed, {k: v for k, v in counts.items() if v}


def _parse_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    text_sentinels: Sequence[str] = (),
    numeric_sentinels: Sequence[float] = (),
    sentinels: dict[str, int],
) -> pd.DataFrame:
    """Strict-parse several columns in place of their text, accumulating sentinel counts."""
    out = frame.copy()
    for col in columns:
        out[col], counts = to_numeric_strict(
            frame[col], text_sentinels=text_sentinels, numeric_sentinels=numeric_sentinels
        )
        for code, n in counts.items():
            sentinels[f"{col}: {code}"] = sentinels.get(f"{col}: {code}", 0) + n
    return out


def _require(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{source}: expected column(s) {missing} not found; layout changed?")


def _month_start(year: pd.Series, month: pd.Series) -> pd.Series:
    """Build month-start timestamps from integer-valued year and month columns."""
    return pd.to_datetime(
        pd.DataFrame({"year": year.astype("int64"), "month": month.astype("int64"), "day": 1})
    )


def _zip_member(archive: zipfile.ZipFile, basename: str) -> zipfile.ZipInfo:
    """Return the single member called ``basename`` (case-insensitive), skipping macOS junk."""
    hits = [
        info
        for info in archive.infolist()
        if not info.is_dir()
        and "__MACOSX" not in info.filename
        and Path(info.filename).name.lower() == basename.lower()
    ]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one member named {basename!r}, found {len(hits)}")
    return hits[0]


def _member_note(info: zipfile.ZipInfo) -> str:
    stamp = "{:04d}-{:02d}-{:02d} {:02d}:{:02d}".format(*info.date_time[:5])
    return f"member {info.filename} ({info.file_size:,} bytes uncompressed, dated {stamp})"


def _read_csv_member(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, notes: list[str], **kwargs: object
) -> pd.DataFrame:
    """Stream a ZIP member as UTF-8 (BOM-tolerant); fall back to Latin-1 and say so in notes.

    Streaming avoids holding the compressed and decoded copies of large members (FAOSTAT:
    ~545 MB) in memory. Latin-1 decodes any byte sequence, so the fallback never fails; the
    note makes the choice visible, and Step 3.3 checks country names for mojibake.
    """
    try:
        with archive.open(info) as handle:
            frame = pd.read_csv(handle, encoding="utf-8-sig", **kwargs)  # type: ignore[call-overload]
        encoding = "utf-8"
    except UnicodeDecodeError:
        with archive.open(info) as handle:
            frame = pd.read_csv(handle, encoding="latin-1", **kwargs)  # type: ignore[call-overload]
        encoding = "latin-1 (not valid UTF-8)"
    notes.append(f"{Path(info.filename).name}: decoded as {encoding}")
    return frame  # type: ignore[no-any-return]


def _path(entry: ManifestEntry, root: Path) -> Path:
    return root / entry.file


def _single(entries: list[ManifestEntry], dataset: str) -> ManifestEntry:
    if len(entries) != 1:
        raise ValueError(f"{dataset}: expected 1 manifest entry, found {len(entries)}")
    return entries[0]


# ---------------------------------------------------------------- global monthly series
def read_gistemp(entries: list[ManifestEntry], root: Path = PROJECT_ROOT) -> list[RawTable]:
    """GISTEMP v4: title line, then wide Year x Jan..Dec; ``***`` = missing."""
    entry = _single(entries, "gistemp")
    wide = pd.read_csv(_path(entry, root), skiprows=1, dtype=str, keep_default_na=False)
    _require(wide, ["Year", *MONTHS], entry.file)
    sentinels: dict[str, int] = {}
    long = wide.melt(id_vars="Year", value_vars=list(MONTHS), var_name="month_name")
    long = _parse_columns(long, ["value"], text_sentinels=["***"], sentinels=sentinels)
    long["year"] = pd.to_numeric(long["Year"], errors="raise").astype("int64")
    long["month"] = long["month_name"].map({m: i + 1 for i, m in enumerate(MONTHS)})
    long["date"] = _month_start(long["year"], long["month"])
    frame = (
        long.rename(columns={"value": "anomaly"})[["date", "year", "month", "anomaly"]]
        .sort_values("date")
        .reset_index(drop=True)
    )
    return [
        RawTable(
            "gistemp_v4_global_monthly",
            "monthly",
            frame,
            [entry.file],
            time_col="date",
            time_freq="M",
            value_cols=["anomaly"],
            key=["date"],
            sentinels=sentinels,
            notes=["J-D, D-N and seasonal columns not read (recomputed in Phase 4)"],
        )
    ]


def read_hadcrut5(entries: list[ManifestEntry], root: Path = PROJECT_ROOT) -> list[RawTable]:
    """HadCRUT5 summary series: ``Time`` (YYYY-MM), anomaly and 95% limits."""
    entry = _single(entries, "hadcrut5")
    raw = pd.read_csv(_path(entry, root), dtype=str, keep_default_na=False)
    names = {
        "Time": "date",
        "Anomaly (deg C)": "anomaly",
        "Lower confidence limit (2.5%)": "lower_2p5",
        "Upper confidence limit (97.5%)": "upper_97p5",
    }
    _require(raw, names, entry.file)
    frame = raw.rename(columns=names)[list(names.values())]
    sentinels: dict[str, int] = {}
    frame = _parse_columns(frame, ["anomaly", "lower_2p5", "upper_97p5"], sentinels=sentinels)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m")
    return [
        RawTable(
            "hadcrut5_global_monthly",
            "monthly",
            frame,
            [entry.file],
            time_col="date",
            time_freq="M",
            value_cols=["anomaly"],
            key=["date"],
            sentinels=sentinels,
        )
    ]


@dataclass
class BerkeleyFile:
    """Parsed Berkeley Earth text file: data rows per section plus header facts."""

    sections: dict[str, list[list[str]]]
    header: dict[str, str]


_BERKELEY_HEADER = {
    "region": re.compile(r"for the region:\s*(?:%\s*)*([^\s%][^\n]*)"),
    "area_km2": re.compile(r"Area:\s*([\d.]+)\s*km"),
    "n_stations": re.compile(r"number of temperature stations:\s*(\d+)"),
    "lat_range": re.compile(r"Latitude Range:\s*([-\d.]+\s*to\s*[-\d.]+)"),
    "lon_range": re.compile(r"Longitude Range:\s*([-\d.]+\s*to\s*[-\d.]+)"),
    "analysis_run": re.compile(r"analysis was run on\s*(.+)"),
    "baseline_abs_c": re.compile(r"absolute temperature \(C\):\s*([-\d.]+)"),
}


def parse_berkeley(path: Path) -> BerkeleyFile:
    """Split a Berkeley ``*.txt`` file into sections of 12-field data rows.

    The global land-ocean file holds two tables, announced by comment lines ending in
    "... Inferred from Air Temperatures" / "... from Water Temperatures"; country files
    hold one (``land``). Header comments are Latin-1 (e.g. an en dash in the citation).
    """
    sections: dict[str, list[list[str]]] = {}
    current = "land"
    comments: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="latin-1").splitlines(), start=1):
        if line.startswith("%"):
            comments.append(line)
            if "Inferred from Air Temperatures" in line:
                current = "air_over_sea_ice"
            elif "Inferred from Water Temperatures" in line:
                current = "water_under_sea_ice"
            continue
        fields = line.split()
        if not fields:
            continue
        if len(fields) != BERKELEY_COLUMNS:
            raise ValueError(f"{path.name}:{lineno}: {len(fields)} fields, expected 12")
        sections.setdefault(current, []).append(fields)
    text = "\n".join(comments)
    header = {k: m.group(1).strip() for k, rx in _BERKELEY_HEADER.items() if (m := rx.search(text))}
    return BerkeleyFile(sections, header)


def _berkeley_frame(rows: list[list[str]], sentinels: dict[str, int]) -> pd.DataFrame:
    """Keep year, month and the monthly anomaly/uncertainty; NaN-padded moving means dropped."""
    raw = pd.DataFrame([r[:4] for r in rows], columns=["year", "month", "anomaly", "uncertainty"])
    frame = _parse_columns(
        raw,
        ["year", "month", "anomaly", "uncertainty"],
        text_sentinels=["NaN"],
        sentinels=sentinels,
    )
    frame["date"] = _month_start(frame["year"], frame["month"])
    frame[["year", "month"]] = frame[["year", "month"]].astype("int64")
    return frame[["date", "year", "month", "anomaly", "uncertainty"]]


def read_berkeley_global(entries: list[ManifestEntry], root: Path = PROJECT_ROOT) -> list[RawTable]:
    """Berkeley Earth land-ocean: two tables (sea-ice treatment); DV1 uses the air table."""
    entry = _single(entries, "berkeley_global")
    parsed = parse_berkeley(_path(entry, root))
    expected = {"air_over_sea_ice", "water_under_sea_ice"}
    if set(parsed.sections) != expected:
        raise ValueError(f"{entry.file}: sections {sorted(parsed.sections)}, expected {expected}")
    tables = []
    for name in sorted(expected):
        sentinels: dict[str, int] = {}
        tables.append(
            RawTable(
                "berkeley_global_monthly",
                name,
                _berkeley_frame(parsed.sections[name], sentinels),
                [entry.file],
                time_col="date",
                time_freq="M",
                value_cols=["anomaly"],
                key=["date"],
                sentinels=sentinels,
                notes=["moving-average columns (annual .. twenty-year) not read"],
            )
        )
    return tables


def read_berkeley_country(
    entries: list[ManifestEntry], root: Path = PROJECT_ROOT
) -> list[RawTable]:
    """Berkeley Earth country files (ISO3 from the manifest) + one header row per file."""
    frames, headers = [], []
    sentinels: dict[str, int] = {}
    for entry in sorted(entries, key=lambda e: e.file):
        iso3 = entry.request.get("iso3")
        if not iso3:
            raise ValueError(f"{entry.file}: manifest entry has no request.iso3")
        parsed = parse_berkeley(_path(entry, root))
        if set(parsed.sections) != {"land"}:
            raise ValueError(f"{entry.file}: unexpected sections {sorted(parsed.sections)}")
        frame = _berkeley_frame(parsed.sections["land"], sentinels)
        frame.insert(0, "iso3", iso3)
        frames.append(frame)
        valid = frame.dropna(subset=["anomaly"])["date"]
        headers.append(
            {
                "iso3": iso3,
                "file": Path(entry.file).name,
                **parsed.header,
                "first_valid": valid.min(),
                "last_valid": valid.max(),
                "last_row": frame["date"].max(),
            }
        )
    files = [e.file for e in entries]
    header = pd.DataFrame(headers)
    for col in ("area_km2", "n_stations", "baseline_abs_c"):
        if col in header:
            header[col] = pd.to_numeric(header[col], errors="raise")
    return [
        RawTable(
            "berkeley_country_monthly",
            "monthly",
            pd.concat(frames, ignore_index=True),
            files,
            time_col="date",
            entity_col="iso3",
            time_freq="M",
            value_cols=["anomaly"],
            key=["iso3", "date"],
            sentinels=sentinels,
            notes=[f"{len(entries)} files; moving-average columns not read"],
        ),
        RawTable(
            "berkeley_country_monthly",
            "file_headers",
            header,
            files,
            entity_col="iso3",
            key=["iso3"],
            notes=["one row per file, parsed from the comment header"],
        ),
    ]


def read_noaa_co2(entries: list[ManifestEntry], root: Path = PROJECT_ROOT) -> list[RawTable]:
    """NOAA GML Mauna Loa monthly CO2; ``#`` comments; one missing code per column."""
    entry = _single(entries, "noaa_co2")
    raw = pd.read_csv(_path(entry, root), comment="#", dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]
    codes = {"average": -99.99, "deseasonalized": -99.99, "ndays": -1, "sdev": -9.99, "unc": -0.99}
    _require(raw, ["year", "month", "decimal date", *codes], entry.file)
    sentinels: dict[str, int] = {}
    frame = _parse_columns(raw, ["year", "month", "decimal date"], sentinels=sentinels)
    for col, code in codes.items():
        frame = _parse_columns(frame, [col], numeric_sentinels=[code], sentinels=sentinels)
    frame["date"] = _month_start(frame["year"], frame["month"])
    frame[["year", "month"]] = frame[["year", "month"]].astype("int64")
    return [
        RawTable(
            "noaa_co2_mlo_monthly",
            "monthly",
            frame[["date", "year", "month", "decimal date", *codes]],
            [entry.file],
            time_col="date",
            time_freq="M",
            value_cols=["average", "deseasonalized"],
            key=["date"],
            sentinels=sentinels,
        )
    ]


def read_oni(entries: list[ManifestEntry], root: Path = PROJECT_ROOT) -> list[RawTable]:
    """NOAA CPC ONI: ``SEAS YR TOTAL ANOM``; each season dated at its centre month."""
    entry = _single(entries, "noaa_oni")
    raw = pd.read_csv(_path(entry, root), sep=r"\s+", dtype=str, keep_default_na=False)
    _require(raw, ["SEAS", "YR", "TOTAL", "ANOM"], entry.file)
    unknown = sorted(set(raw["SEAS"]) - set(ONI_SEASONS))
    if unknown:
        raise ValueError(f"{entry.file}: unknown season codes {unknown}")
    sentinels: dict[str, int] = {}
    frame = _parse_columns(raw, ["YR", "TOTAL", "ANOM"], sentinels=sentinels)
    frame["month"] = frame["SEAS"].map({s: i + 1 for i, s in enumerate(ONI_SEASONS)})
    frame["date"] = _month_start(frame["YR"], frame["month"])
    return [
        RawTable(
            "noaa_oni",
            "monthly",
            frame[["date", "SEAS", "YR", "TOTAL", "ANOM"]],
            [entry.file],
            time_col="date",
            time_freq="M",
            value_cols=["ANOM"],
            key=["date"],
            sentinels=sentinels,
            notes=["date = centre month of the 3-month season (DJF -> Jan of YR, NDJ -> Dec)"],
        )
    ]


# ---------------------------------------------------------------- country panels
def read_owid(entries: list[ManifestEntry], root: Path = PROJECT_ROOT) -> list[RawTable]:
    """OWID CO2 data: long country-year; aggregates have no (or ``OWID_*``) ``iso_code``."""
    entry = _single(entries, "owid")
    frame = pd.read_csv(_path(entry, root), dtype={"country": str, "iso_code": str})
    _require(frame, ["country", "year", "iso_code", "co2", "co2_per_capita"], entry.file)
    return [
        RawTable(
            "owid_co2",
            "country_year",
            frame,
            [entry.file],
            time_col="year",
            entity_col="iso_code",
            time_freq="Y",
            value_cols=["co2", "co2_per_capita"],
            key=["country", "year"],
        )
    ]


FAOSTAT_REQUIRED = ("Area Code", "Area", "Item Code", "Element Code", "Element", "Year", "Unit")
FAOSTAT_CATEGORIES = (
    "Area Code",
    "Area Code (M49)",
    "Area",
    "Item Code",
    "Item Code (CPC)",
    "Item",
    "Element Code",
    "Element",
    "Unit",
    "Flag",
    "Note",
)


def read_faostat(entries: list[ManifestEntry], root: Path = PROJECT_ROOT) -> list[RawTable]:
    """FAOSTAT normalized bulk ZIP: main long table (+ the Flags legend)."""
    entry = _single(entries, "faostat")
    notes: list[str] = []
    with zipfile.ZipFile(_path(entry, root)) as archive:
        mains = [
            i
            for i in archive.infolist()
            if i.filename.lower().endswith(".csv") and "all_data" in i.filename.lower()
        ]
        if len(mains) != 1:
            raise ValueError(
                f"{entry.file}: expected one *All_Data*.csv member, found {len(mains)}"
            )
        notes += [_member_note(i) for i in archive.infolist()]
        with archive.open(mains[0]) as handle:
            head = pd.read_csv(handle, nrows=0, encoding="latin-1").columns
        dtypes = {c: "category" for c in FAOSTAT_CATEGORIES if c in head}
        frame = _read_csv_member(
            archive, mains[0], notes, dtype={**dtypes, "Value": str}, keep_default_na=False
        )
        flags_info = next(
            (i for i in archive.infolist() if i.filename.lower().endswith("flags.csv")), None
        )
        legend = _read_csv_member(archive, flags_info, [], dtype=str) if flags_info else None
    _require(frame, [*FAOSTAT_REQUIRED, "Value"], entry.file)
    sentinels: dict[str, int] = {}
    frame = _parse_columns(frame, ["Value"], sentinels=sentinels)
    frame["Year"] = pd.to_numeric(frame["Year"], errors="raise").astype("int64")
    if "Year Code" in frame:
        frame = frame.drop(columns="Year Code")
        notes.append("'Year Code' dropped (duplicates 'Year' in annual data)")
    tables = [
        RawTable(
            "faostat_crops_livestock",
            "production",
            frame,
            [entry.file],
            time_col="Year",
            entity_col="Area Code",
            time_freq="Y",
            value_cols=["Value"],
            key=["Area Code", "Item Code", "Element Code", "Year"],
            sentinels=sentinels,
            notes=notes,
        )
    ]
    if legend is not None:
        tables.append(RawTable("faostat_crops_livestock", "flags_legend", legend, [entry.file]))
    return tables


def _melt_years(
    wide: pd.DataFrame, id_vars: list[str], source: str, sentinels: dict[str, int]
) -> pd.DataFrame:
    """Wide year columns (``"1960"``, ...) -> long ``year``/``value`` with strict parsing."""
    years = [c for c in wide.columns if YEAR_COLUMN.match(str(c))]
    if not years:
        raise ValueError(f"{source}: no year columns found")
    long = wide.melt(id_vars=id_vars, value_vars=years, var_name="year")
    long = _parse_columns(long, ["value"], sentinels=sentinels)
    long["year"] = long["year"].astype("int64")
    return long


def read_wdi(entries: list[ManifestEntry], root: Path = PROJECT_ROOT) -> list[RawTable]:
    """WDI bulk ZIP: the five configured indicators (long) + the country classification."""
    entry = _single(entries, "wdi")
    indicators = set(load_datasets()["wdi"]["indicators"].values())
    notes: list[str] = []
    with zipfile.ZipFile(_path(entry, root)) as archive:
        data_info = _zip_member(archive, "WDICSV.csv")
        country_info = _zip_member(archive, "WDICountry.csv")
        notes += [_member_note(i) for i in archive.infolist()]
        parts = []
        with archive.open(data_info) as handle:
            reader = pd.read_csv(
                handle, dtype=str, keep_default_na=False, chunksize=50_000, encoding="utf-8-sig"
            )
            for chunk in reader:
                _require(chunk, ["Country Code", "Indicator Code"], data_info.filename)
                parts.append(chunk[chunk["Indicator Code"].isin(indicators)])
        country = _read_csv_member(archive, country_info, notes, dtype=str, keep_default_na=False)
    wide = pd.concat(parts, ignore_index=True)
    found = set(wide["Indicator Code"])
    if found != indicators:
        raise ValueError(f"{entry.file}: indicators missing from WDICSV.csv: {indicators - found}")
    sentinels: dict[str, int] = {}
    ids = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
    _require(wide, ids, data_info.filename)
    long = _melt_years(wide, ids, data_info.filename, sentinels)
    _require(country, ["Country Code", "Region", "Income Group"], country_info.filename)
    return [
        RawTable(
            "wdi",
            "indicators",
            long,
            [entry.file],
            time_col="year",
            entity_col="Country Code",
            time_freq="Y",
            value_cols=["value"],
            key=["Country Code", "Indicator Code", "year"],
            sentinels=sentinels,
            notes=[*notes, f"filtered to the {len(indicators)} configured indicators"],
        ),
        RawTable(
            "wdi",
            "country_classification",
            country,
            [entry.file],
            entity_col="Country Code",
            key=["Country Code"],
            notes=["empty Region = aggregate (World, income groups, regions)"],
        ),
    ]


def read_emdat(entries: list[ManifestEntry], root: Path = PROJECT_ROOT) -> list[RawTable]:
    """EM-DAT export (.xlsx, sheet ``EM-DAT Data``): one row per event x country."""
    entry = _single(entries, "emdat")
    path = _path(entry, root)
    with pd.ExcelFile(path, engine="openpyxl") as book:
        sheets = list(book.sheet_names)
        frame = pd.read_excel(book, sheet_name="EM-DAT Data")
    _require(
        frame,
        ["DisNo.", "ISO", "Disaster Subgroup", "Start Year", "Total Deaths", "Total Affected"],
        entry.file,
    )
    return [
        RawTable(
            "emdat",
            "events",
            frame,
            [entry.file],
            time_col="Start Year",
            entity_col="ISO",
            time_freq="Y",
            value_cols=["Total Deaths", "Total Affected"],
            key=["DisNo."],
            notes=[f"sheets: {sheets}; licence: aggregates only, never publish records"],
        )
    ]


def read_nd_gain(entries: list[ManifestEntry], root: Path = PROJECT_ROOT) -> list[RawTable]:
    """ND-GAIN ZIP: vulnerability and readiness scores, wide by year -> long."""
    entry = _single(entries, "nd_gain")
    tables = []
    with zipfile.ZipFile(_path(entry, root)) as archive:
        for name in ("vulnerability", "readiness"):
            info = _zip_member(archive, f"{name}.csv")
            notes = [_member_note(info)]
            wide = _read_csv_member(archive, info, notes, dtype=str)
            _require(wide, ["ISO3", "Name"], info.filename)
            sentinels: dict[str, int] = {}
            long = _melt_years(wide.fillna(""), ["ISO3", "Name"], info.filename, sentinels)
            tables.append(
                RawTable(
                    "nd_gain",
                    name,
                    long,
                    [entry.file],
                    time_col="year",
                    entity_col="ISO3",
                    time_freq="Y",
                    value_cols=["value"],
                    key=["ISO3", "year"],
                    sentinels=sentinels,
                    notes=notes,
                )
            )
    return tables


Reader = Callable[[list[ManifestEntry], Path], list[RawTable]]

READERS: dict[str, Reader] = {
    "gistemp_v4_global_monthly": read_gistemp,
    "hadcrut5_global_monthly": read_hadcrut5,
    "berkeley_global_monthly": read_berkeley_global,
    "berkeley_country_monthly": read_berkeley_country,
    "noaa_co2_mlo_monthly": read_noaa_co2,
    "noaa_oni": read_oni,
    "owid_co2": read_owid,
    "faostat_crops_livestock": read_faostat,
    "wdi": read_wdi,
    "emdat": read_emdat,
    "nd_gain": read_nd_gain,
}


def read_dataset(
    name: str, manifest: list[ManifestEntry], root: Path = PROJECT_ROOT
) -> list[RawTable]:
    """Read every table of one dataset from the files registered for it in the manifest."""
    entries = [e for e in manifest if e.dataset == name]
    if not entries:
        raise FileNotFoundError(f"{name}: no files registered in the manifest")
    logger.info("%s: reading %d file(s)", name, len(entries))
    return READERS[name](entries, root)
