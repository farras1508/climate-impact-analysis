# Data Dictionary

Input side of the pipeline: raw file -> variable. Output tables (ABTs) are specified in
`config/abt_schemas.yaml`. Units are given **as delivered** and **as used** after Phase 4
conversion. "Observed" = seen in the downloaded file during Phase 2; "(confirm in Phase 3)" =
from documentation or expectation, not yet inspected.

## 1. Raw files

| Dataset (manifest name) | File in `data/raw/` | Layout (observed) | Native baseline / version |
|---|---|---|---|
| gistemp_v4_global_monthly | `gistemp/GLB.Ts+dSST.csv` | Line 1 title ("Land-Ocean: Global Means"), line 2 header; wide: `Year, Jan..Dec, J-D, D-N, DJF, MAM, JJA, SON` | Anomaly vs 1951-1980; GISTEMP v4 (GHCN v4 + ERSST v5) |
| hadcrut5_global_monthly | `hadcrut5/global_monthly.csv` | Long: `Time` (YYYY-MM), `Anomaly (deg C)`, `Lower/Upper confidence limit (2.5%/97.5%)` | Anomaly vs 1961-1990; HadCRUT 5.2.0.0 |
| berkeley_global_monthly | `berkeley/Land_and_Ocean_complete.txt` | `%` comment header; whitespace columns: year, month, monthly anomaly, uncertainty, then moving averages (NaN-padded); **two tables** (air temperature over sea ice; water temperature under sea ice) | Anomaly vs Jan 1951-Dec 1980; classic product, ends 2024-12 |
| berkeley_country_monthly | `berkeley/country/<stem>-TAVG-Trend.txt` (228; ISO3 map in `config/berkeley_countries.yaml`) | Same layout as global, **land only**; absolute climatology in header | Anomaly vs 1951-1980; 195 files end 2020-12, 33 end 2016-05 |
| noaa_co2_mlo_monthly | `noaa/co2_mm_mlo.csv` | `#` comment lines; columns `year, month, decimal date, average, deseasonalized, ndays, sdev, unc`; starts 1958-03 | NOAA GML Mauna Loa |
| noaa_oni | `noaa/oni.ascii.txt` | Whitespace: `SEAS YR TOTAL ANOM`; `SEAS` = 3-letter season (DJF, JFM, ..., NDJ) | ONI (Nino 3.4, 3-month running mean, centred) |
| owid_co2 | `owid/owid-co2-data.csv` | Long: `country, year, iso_code, ...` (79+ columns) | OWID CO2 dataset |
| faostat_crops_livestock | `faostat/Production_Crops_Livestock_E_All_Data_(Normalized).zip` | ZIP: main CSV (545 MB uncompressed) + `AreaCodes`, `Elements`, `Flags`, `ItemCodes` CSVs; long ("normalized") layout (confirm columns in Phase 3) | Bulk file of 2025-12-31 |
| wdi | `wdi/WDI_CSV.zip` | ZIP: `WDICSV.csv` (data; wide by year, confirm in Phase 3), `WDICountry.csv` (Region, Income Group), `WDISeries.csv`, footnotes | WDI bulk, browser download 2026-11-16 |
| emdat | `emdat/emdat_export.xlsx` | Sheets `EM-DAT Data` (18,086 x 47, observed) and `EM-DAT Info`; one row per event x country | EM-DAT last update 2026-11-10; natural, all years incl. historic |
| nd_gain | `nd_gain/nd_gain_country_index_2026_release_data_v2.zip` | ZIP: `vulnerability/vulnerability.csv`, `readiness/readiness.csv`, `gain/gain.csv`, indicator folders, codebook PDF; macOS artefacts (`__MACOSX`, `.DS_Store`) to skip | ND-GAIN 2026 release (data v2) |
| era5_monthly_single_levels | `era5/era5_monthly_{first}_{last}.zip` (6 chunks) | ZIP of 2 NetCDF: `..._stepType-avgua.nc` (t2m), `..._stepType-avgad.nc` (tp); dims `valid_time x latitude(721) x longitude(1440)`; coord `expver` | ERA5 monthly averaged reanalysis (CDS) |
| natural_earth_admin0 | via regionmask cache (not in `data/raw/`, DEC-060) | `natural_earth_v5_1_2.countries_50`: 242 regions | Natural Earth v5.1.2, regionmask 0.13.0 |

## 2. Variables

| Variable | Source field | Description | Unit delivered -> used | Type | Grain | Valid range (used) | Missing code(s) in raw | Parsing / conversion notes |
|---|---|---|---|---|---|---|---|---|
| `temp_anom_gistemp` | GISTEMP `Jan..Dec` | Global land-ocean temperature anomaly | degC vs 1951-1980 -> degC vs 1991-2020 | float | global-month | -2.5 to 3.0 | `***` | Melt wide -> long; `***` -> NaN **at read**; ignore J-D/D-N/seasonal columns (recomputed); rebase per calendar month |
| `temp_anom_hadcrut5` | HadCRUT5 `Anomaly (deg C)` | Global temperature anomaly (ensemble mean) | degC vs 1961-1990 -> degC vs 1991-2020 | float | global-month | -2.5 to 3.0 | none observed | Parse `Time` as YYYY-MM; keep 2.5%/97.5% limits as `hadcrut5_lower`, `hadcrut5_upper` (uncertainty) |
| `temp_anom_berkeley` (global) | Berkeley monthly anomaly | Global land-ocean anomaly | degC vs 1951-1980 -> degC vs 1991-2020 | float | global-month | -2.5 to 3.0 | `NaN` | Use the **air-temperature-over-sea-ice** table (plan DV1); identify tables from header text, assert which is which |
| `temp_anom_berkeley` (country) | Berkeley country monthly anomaly | Country land-surface anomaly | degC vs 1951-1980 -> degC vs 1991-2020 | float | country-month | -15 to 15 (monthly; DEC-064) | `NaN` | ISO3 via `berkeley_countries.yaml`; H1c replication only for files reaching 2020-12 (DV2) |
| `co2_ppm` | NOAA `average` | Monthly mean CO2, Mauna Loa | ppm | float | global-month | 300 to 500 | `-99.99` (confirm in Phase 3) | Comment lines start with `#` |
| `co2_ppm_deseason` | NOAA `deseasonalized` | CO2 with seasonal cycle removed | ppm | float | global-month | 300 to 500 | `-99.99` (confirm) | Used in H2 (DEC-028); `ndays=-1`, `sdev=-9.99`, `unc=-0.99` are ancillary missing codes |
| `oni` | ONI `ANOM` | ENSO index | degC | float | global-month | -3.5 to 3.5 | none observed | Assign each 3-month season to its **centre month** (DJF->Jan, JFM->Feb, ..., NDJ->Dec); `TOTAL` is absolute SST, not used |
| `t2m_c` | ERA5 `t2m` (member `avgua`) | 2 m air temperature | K -> degC | float | grid-month | -90 to 60 (grid) | none | `t2m - 273.15`; `valid_time` at 00:00 |
| `tp_mm_month` | ERA5 `tp` (member `avgad`) | Total precipitation | m/day (monthly mean of daily accumulation) -> mm/month | float | grid-month | >= 0 | none | `tp * 1000 * days_in_month`; **`valid_time` stamped 06:00 -> normalize to month start before joining with t2m** |
| ERA5 grid | `latitude`, `longitude`, `expver` | 0.25 deg grid | degrees | float | grid | lat 90 -> -90 (descending); lon 0 -> 359.75 | - | Handle 0-360 longitudes in masking; area weights cos(lat); check `expver` (final ERA5 vs ERA5T) in Phase 3 |
| `yield_kg_ha` | FAOSTAT element 5412 (or 5419 hg/ha) | Crop yield | kg/ha (hg/ha / 10) | float | country-year-crop | 1 to 30000 | absent rows | Items 15, 27, 56, 236 by **code**; assert single unit after conversion; keep `Flag` |
| `area_harvested_ha` | FAOSTAT element 5312 | Harvested area | ha | float | country-year-crop | >= 0 | absent rows | Used for H3 weights and sample threshold |
| `production_t` | FAOSTAT element 5510 | Production | t | float | country-year-crop | >= 0 | absent rows | `yield_computed = production / area` cross-check |
| FAOSTAT area keys | `AreaCodes` file | FAO area -> ISO3 / M49 | code | str | country | - | - | Drop regional aggregates and historical entities per crosswalk (Step 4.6) |
| `population` | WDI `SP.POP.TOTL` | Population | persons | float | country-year | 1e3 to 2e9 | blank | Lagged 1 year for H4 controls |
| `gdp_pc_ppp` | WDI `NY.GDP.PCAP.PP.KD` | GDP per capita, PPP, constant international $ | int$ | float | country-year | 100 to 250000 | blank | Lagged 1 year for H4 controls |
| `agri_share_gdp` | WDI `NV.AGR.TOTL.ZS` | Agriculture value added | % of GDP | float | country-year | 0 to 100 | blank | H5 feature: mean of last 10 years |
| `urban_share`, `land_area_km2` | WDI `SP.URB.TOTL.IN.ZS`, `AG.LND.TOTL.K2` | Urban population share; land area | %; km2 | float | country-year | 0-100; > 0 | blank | Profiles / exploratory |
| `wb_region`, `wb_income_group` | `WDICountry.csv` Region, Income Group | Fixed reference classification (DEC-035, DEC-054) | code | str | country | - | blank = aggregate | Rows with empty Region are aggregates -> excluded |
| disaster events | EM-DAT `DisNo., ISO, Disaster Subgroup, Start Year, Total Deaths, Historic` | Natural disaster event x country | count / persons | mixed | event-country | Start Year 1900-2026 (observed) | blank | Climate = Hydrological, Meteorological, Climatological; negative control = Geophysical; drop Biological, Extra-terrestrial; count by `Start Year`; structural zeros inside H4 window (DEC-034); read with `pd.read_excel` |
| `ndgain_vulnerability`, `ndgain_readiness` | ND-GAIN `vulnerability.csv`, `readiness.csv` | Vulnerability and readiness scores | 0-1 | float | country (latest year) | 0 to 1 | blank (confirm) | Latest year present for all countries; layout (wide by year?) confirm in Phase 3 |
| `co2_per_capita` | OWID `co2_per_capita` | Territorial CO2 per person | t/person | float | country-year | >= 0 | blank | Profiles only (not an H5 feature); aggregates have blank or `OWID_*` `iso_code` -> excluded |

## 3. Key harmonization rules (summary; details in plan section 3)

- Country key: ISO 3166-1 alpha-3 everywhere; aggregates and historical entities removed in Phase 4.
- Time key: month-start dates; annual values only from 12 valid months.
- Every sentinel code listed above becomes NaN **when the file is read**, never later.
- Every unit conversion has a unit test (Phase 11 strategy, written in Phase 4).
