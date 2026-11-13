# Analysis Plan (Pre-registration)

**Status:** FROZEN — `prereg-v1`, 2026-11-13. Sections 1-6 change only through Section 7 (Deviations log).
**Rule:** No project data has been downloaded or inspected while writing this plan.
Changes after freezing are recorded in Section 7 (Deviations log) with date and reason.

---

## 1. Purpose, audiences and decision context

### 1.1 Personas

| ID | Persona | Decision they make | Questions served | What they need | Most harmful error | Uncertainty format |
|----|---------|--------------------|------------------|----------------|--------------------|--------------------|
| P0 | Technical reviewer (hiring manager, data scientist) | Is this work rigorous and reproducible? | All | Pre-registration, tests, `make all`, honest limitations | Hidden analytical flexibility; irreproducible numbers | Full statistical detail (CIs, diagnostics, robustness) |
| P1 | Policy analyst (government, NGO, adaptation finance) | Where to direct adaptation attention and funds across countries | Q1c, Q5 (Q3, Q4 as inputs) | Country risk segments and profiles; country warming trends | **False negative:** a high-hazard, low-capacity country labelled low risk -> under-funded | Segment-membership stability (bootstrap); rank intervals, never bare ranks |
| P2 | Agricultural risk analyst (insurer, commodity trader, agriculture ministry) | Pricing and hedging yield risk under warm growing seasons | Q3 | % yield change per +1 °C by crop; predictions with intervals; exposure ranking | **Underestimating losses in hot years** (tail risk) | Prediction intervals with stated, tested coverage; effect CIs |
| P3 | Informed public / journalist | What to report about warming, drivers and impacts | Q1, Q2, Q4 | A few robust headline numbers; clear charts | **Overclaiming:** causal or event-attribution language the design cannot support; false precision | IPCC AR6 calibrated language; ranges, not point values |

**Out of scope audiences:** farm-level decision makers (country-level models do not apply to
individual farms — ecological fallacy, see 1.4) and users needing future climate projections
(requires CMIP6 models; see Charter §1.3).

### 1.2 Traceability: personas -> questions

| Question | P0 | P1 | P2 | P3 |
|----------|:--:|:--:|:--:|:--:|
| Q1 Signal (global & country warming) | ● | ● (Q1c) | | ● |
| Q2 Drivers (CO₂, ENSO) | ● | | | ● |
| Q3 Crop-yield sensitivity | ● | ○ | ● | |
| Q4 Disaster frequency | ● | ○ | | ● |
| Q5 Risk segmentation | ● | ● | | |

● primary user · ○ uses as input. Every question serves at least one external persona (P1–P3),
and every persona is served by at least one question.

### 1.3 Requirements derived from personas

These are binding on later phases; each will be checked at the Phase 9 and Phase 10 gates.

| ID | Requirement | Source | Phase where enforced |
|----|-------------|--------|----------------------|
| R1 | Every country-level estimate is shown with its uncertainty; country rankings are reported with bootstrap **rank intervals**, never bare ranks | P1 | 6, 8, 10 |
| R2 | Q5 segmentation reports cluster **stability** (bootstrap ARI) and the share of countries whose segment changes across bootstrap resamples | P1 | 8, 9 |
| R3 | The yield model is evaluated **separately on hot growing seasons** (top decile of growing-season temperature anomaly in the test set), not only on average error | P2 | 9 |
| R4 | Yield prediction intervals report **empirical coverage** on the locked test set against their nominal level (e.g. 90%) | P2 | 9 |
| R5 | Q3 effects are expressed as **% yield change per +1 °C** with 95% CI, so they are directly comparable with published benchmarks | P2, P0 | 6, 10 |
| R6 | Public-facing text uses IPCC AR6 calibrated likelihood terms via a pre-stated mapping (to be defined in Phase 10); the mapping is a stated convention, not a claim of IPCC equivalence | P3 | 10 |
| R7 | Association language only ("is associated with", "consistent with"); no event attribution; causal wording only where the design supports it and with assumptions stated | P3, Charter §1.3 | 10 |
| R8 | Countries with few grid cells or heavy FAOSTAT imputation are **visibly flagged** wherever their values appear | P1, P3 | 4, 10 |

### 1.4 Misuse risks and mitigations

| Risk | Who might misuse | Mitigation |
|------|------------------|------------|
| Country risk rankings used to stigmatize or as a sole basis for funding | P1 | Rank intervals (R1); segments described as screening tools; data-quality flags (R8) |
| Ecological fallacy: country-level yield response applied to individual farms or regions | P2 | Explicit statement in model card and dashboard; country-level framing throughout |
| Disaster-count association read as "climate change caused disaster X" | P3 | R7; negative-control result shown alongside (DEC-012) |
| Point forecasts quoted without uncertainty | P2, P3 | Intervals always displayed with forecasts (R4, R6) |

### 1.5 Focus countries
**None (DEC-026).** No country is singled out for case-study treatment. Every country receives
an equal, automatically generated profile in the dashboard. This removes the risk of selecting
case studies after seeing which countries show the most striking results.

## 2. Hypothesis register

### 2.0 Conventions for all hypotheses

- **Tiers.** *Confirmatory*: the primary test of each hypothesis; only these support claims.
  *Secondary*: pre-specified robustness checks; they qualify confirmatory results.
  *Exploratory*: anything not listed here; always labelled as exploratory, never claimed.
- **Inference.** Two-sided, alpha = 0.05; 95% CIs reported for every estimate, significant or not.
- **Few clusters.** If a cluster-robust analysis has fewer than 40 clusters (e.g. countries for one
  crop in H3), inference uses the wild cluster bootstrap (Webb six-point weights, 9,999 replications;
  MacKinnon & Webb 2017) instead of analytical cluster-robust SE (Cameron & Miller 2015).
- **Multiple testing.**
  - H1a, H1b, H2: the decision requires every dataset to pass (intersection-union test; Berger 1982), so no further correction is needed.
  - H1c: Benjamini-Hochberg FDR across countries. Robustness: Benjamini-Yekutieli, valid under any dependence.
  - H3: Holm correction across the 4 crops.
  - H4: one primary test.
- **Null results.** Where a SESOI is defined, a non-significant estimate is called "negligible" only if
  its 90% CI lies within +/- SESOI (TOST; Lakens 2017). Otherwise it is "inconclusive".
- **Reporting.** Every pre-registered test is reported, whatever the outcome.
- **Software.** statsmodels, pymannkendall, piecewise-regression, ruptures, pyfixest, linearmodels
  (cross-check), scikit-learn. Versions are pinned in `environment.lock.yml`.
- **Numeric parameters** are stored in `config/config.yaml` (`hypotheses:` block), not hard-coded.

### 2.1 Summary

| ID | H1 (alternative) | Unit | Primary method | Decision rule (supported if...) |
|----|------------------|------|----------------|---------------------------------|
| H1a | GMST has a positive trend, 1961-2025 | Global x year | OLS + Newey-West HAC; modified Mann-Kendall + Sen | slope > 0, HAC 95% CI excludes 0 **and** MMK p < 0.05, in **all 3** datasets |
| H1b | The warming rate increased at a break in 1960-1985 | Global x year, 1880-2025 | Continuous segmented regression (Muggeo) + Davies test | Davies p < 0.05; most recent break CI overlaps 1960-1985; slope increase CI excludes 0, in all 3 datasets |
| H1c | Most countries warmed significantly | Country x year | Per-country modified MK + Sen; BH-FDR | > 50% of countries with q < 0.05 and Sen slope > 0, in ERA5 **and** Berkeley Earth |
| H2 | GMST is positively associated with ln(CO2), controlling for ENSO | Global x month, 1959-2025 | Cointegration test, then DOLS + HAC | Cointegration rejected-null at 5% **and** beta > 0 with CI excluding 0, in all 3 datasets |
| H3 | Yields respond as an inverted U to growing-season temperature | Country x year x crop | Two-way FE + country quadratic trends; Lind-Mehlum U-test | Lind-Mehlum p < 0.05 (Holm across crops); turning point inside the observed range |
| H4 | Climate-related disaster counts rise with local temperature anomalies | Country x year, 1990-2025 | PPML with country & year FE | IRR > 1, CI excludes 1, **and** the geophysical placebo IRR CI includes 1 |
| H5 | Countries form stable climate-risk segments | Country | k-means (k chosen by silhouette); bootstrap stability | Silhouette > 0.25 **and** mean bootstrap ARI > 0.6 |

### 2.2 H1a: Global warming trend

| Field | Specification |
|-------|---------------|
| Estimand | Linear trend of annual global mean surface temperature (GMST) anomaly, in degC/decade, 1961-2025 |
| Data | GISTEMP v4, HadCRUT5, Berkeley Earth. Annual means require 12 valid months. Rebased to 1991-2020 (rebasing does not change a slope) |
| Primary | OLS of anomaly on year; Newey-West HAC SE with maxlags = floor(4(T/100)^(2/9)) |
| Co-primary | Modified Mann-Kendall (Hamed & Rao 1998 autocorrelation correction) and Sen's slope with CI |
| Decision | Intersection-union: all 3 datasets have slope > 0 with HAC CI excluding 0 and MMK p < 0.05 |
| Triangulation (charter success criterion) | Pairwise overlap of the three 95% CIs is reported; non-overlap is itself a finding |
| Plausibility check | A slope outside 0.10-0.30 degC/decade triggers a **data-pipeline audit** before interpretation. This checks the pipeline; it is not a hypothesis |
| Secondary | (i) Start year 1970; (ii) monthly series with HAC; (iii) HadCRUT5 200-member ensemble: distribution of the slope across members (measurement uncertainty) |
| Caveat | A linear trend is a summary statistic, not a physical model; non-linearity is addressed by H1b |

### 2.3 H1b: Change in warming rate

| Field | Specification |
|-------|---------------|
| Estimand | Year of the most recent break in GMST trend slope and the slope change (degC/decade) at it |
| Window | 1880-2025 (all three datasets cover it). Deviation from the framework's common window, because a break in the 1970s leaves too little pre-break data within 1961-2025 (DEC-027) |
| Primary | Continuous piecewise-linear regression (Muggeo 2003). Number of breaks k in {0,1,2,3} selected by BIC. Minimum segment length 15 years |
| Test | Davies (1987) test for the existence of a slope change |
| Decision | Davies p < 0.05; the most recent break's 95% CI overlaps 1960-1985; post-break minus pre-break slope > 0 with CI excluding 0; in all 3 datasets |
| Autocorrelation | Davies and Muggeo assume independent errors. Robustness: moving-block bootstrap (block length 5) for the break-date and slope-change CIs |
| Secondary | ruptures dynamic programming with a piecewise-linear cost and BIC penalty |
| Descriptive only | Pettitt test. It detects a shift in level, and any trending series will trigger it, so it cannot test H1b |
| Exploratory | Heterogeneity by income group and irrigation share; degree-day features (stretch); long-differences estimator comparing ~20-year averages (Burke & Emerick 2016) to contrast short-run weather and long-run climate responses |

### 2.4 H1c: Country-level warming

| Field | Specification |
|-------|---------------|
| Estimand | Proportion of analysed countries with a significant positive trend in annual mean temperature, 1961-2025 |
| Data | Primary: ERA5, area-weighted country means. Replication: Berkeley Earth country series |
| Method | Per-country modified MK + Sen's slope; BH-FDR at q < 0.05 across all countries |
| Decision | > 50% of countries with q < 0.05 and Sen slope > 0, in **both** ERA5 and Berkeley Earth |
| Note on uncertainty | The countries are the whole population, not a sample, so a sampling CI on the proportion is not meaningful. Uncertainty is shown through dataset agreement (ERA5 vs Berkeley) and the FDR method (BH vs BY) |
| Dependence | Neighbouring countries' tests are positively correlated. BH remains valid under positive regression dependence (Benjamini & Yekutieli 2001). Robustness: the BY procedure |
| Secondary | Excluding countries flagged with fewer than 10 grid cells (R8); also report the share with significant **negative** trends |
| Exploratory | Ratio of land warming to global warming rates |

### 2.5 H2: Temperature, CO2 and ENSO

| Field | Specification |
|-------|---------------|
| Estimand | beta in GMST_t = a + beta * ln(CO2_t) + g * ONI_(t-3) + e_t; also reported as beta * ln 2 (degC per CO2 doubling) |
| Data | Monthly, 1959-01 to 2025-12. Mauna Loa **deseasonalized** CO2 (the seasonal cycle is not a monthly-scale forcing). ONI lagged 3 months |
| Step 1: stationarity | ADF and KPSS (with trend) on GMST and ln(CO2) |
| Step 2: cointegration | Engle-Granger test on the residuals (Phillips-Ouliaris as robustness) |
| Step 3: primary estimator | Dynamic OLS (Stock & Watson 1993) with +/- 12 leads and lags of delta ln(CO2); HAC SE |
| Decision | Cointegration test rejects the no-cointegration null at 5% **and** beta > 0 with 95% CI excluding 0, in all 3 datasets |
| If not cointegrated | H2 is reported as **inconclusive as specified**; a first-difference model is reported as secondary |
| Volcanic forcing (robustness) | Re-estimate after excluding the 24 months following El Chichon (1982-04 to 1984-03) and Pinatubo (1991-06 to 1993-05). If beta moves by more than its own SE, the omission of volcanic forcing is flagged |
| Identification check | Compare models using ln(CO2) and a smooth time trend by BIC. If they are indistinguishable, the report states that the data cannot separate CO2 from other smooth trends |
| Secondary | ONI lags 0-6; plain OLS + HAC; each dataset separately |
| Interpretation guardrail | beta * ln 2 is an **empirical transient association** absorbing all forcings correlated with CO2 (other greenhouse gases, aerosols). It is **not** an estimate of TCR or ECS (Charter section 1.3). The IPCC AR6 TCR range is shown only for context |
| Forecast (Q2) | Prediction task; evaluation is pre-specified in section 2.8 |

### 2.6 H3: Crop-yield temperature response

| Field | Specification |
|-------|---------------|
| Estimand | (a) The inverted-U shape of log yield in growing-season (GS) temperature. (b) Production-weighted marginal effect of +1 degC GS temperature, in % yield change (R5) |
| Unit and sample | Country x year, one model per crop (maize, wheat, rice, soybeans). Country-crop pairs need >= 20 years of yield data and mean harvested area >= 10,000 ha |
| Outcome | ln(yield, kg/ha) |
| Regressors | GS mean temperature T, T^2, GS total precipitation P, P^2. Area-weighted country climate (cropland-weighted is a stretch goal) |
| Fixed effects | Country FE + year FE + country-specific quadratic time trends (the Burke, Hsiang & Miguel 2015 specification). The trends absorb country-specific technology growth (risk register) |
| Weights | Primary: mean harvested area, held fixed over time (estimand relevant to global production). Robustness: unweighted. Weighting changes the estimand when effects differ between countries (Solon, Haider & Wooldridge 2015), so both are reported |
| SE | Two-way clustered by country and year (weather shocks are spatially correlated within a year). Robustness: clustered by country only |
| Primary test | Lind & Mehlum (2010) U-test: slope significantly > 0 at the observed minimum T and significantly < 0 at the observed maximum. Holm-adjusted across the 4 crops |
| Decision | Lind-Mehlum Holm-adjusted p < 0.05, and the turning point T* = -b_T / (2 b_T2) lies inside the observed T range, with a Fieller CI |
| Effect (b) | 100 * [exp(b_T + b_T2 * (2*Tbar_i + 1)) - 1] at each country's mean T, then production-weighted. 95% CI from a country-cluster bootstrap |
| SESOI | +/- 1% yield change per degC |
| Benchmark | Zhao et al. (2017): maize -7.4%, wheat -6.0%, rice -3.2%, soybean -3.1% per degC of **global** warming. Our effect is per degC of **local GS** temperature, and land warms faster than the global mean, so the comparison is approximate and the report says so |
| Prior expectation (honest) | Damage concentrates above daily thresholds (about 29-30 degC for maize and soy; Schlenker & Roberts 2009). Country-season averages smooth over those extremes, so the inverted U may be attenuated (aggregation bias) |
| Secondary | (i) Excluding FAOSTAT estimated/imputed values; (ii) country linear trends; (iii) first-difference estimator; (iv) dropping the top 5 producers one at a time (influence); (v) leave-one-region-out |
| Exploratory | Heterogeneity by income group and irrigation share; degree-day features (stretch) |

### 2.7 H4: Disaster frequency

| Field | Specification |
|-------|---------------|
| Estimand | Incidence rate ratio (IRR) of climate-related disaster counts per +1 degC of country annual temperature anomaly |
| Window | Primary 1990-2025. Sensitivity 1980 and 2000 starts (reporting completeness improved over time) |
| Outcome | Count of EM-DAT events in the hydrological, meteorological and climatological subgroups, by start year. Multi-country events count once per affected country |
| Exposure | Country annual mean temperature anomaly (ERA5, 1991-2020 baseline) |
| Controls | ln(GDP per capita, PPP) and ln(population), both **lagged 1 year**, because same-year values can be affected by the disasters themselves |
| Estimator | PPML with country and year FE (Santos Silva & Tenreyro 2006); SE clustered by country. Countries with all-zero counts drop out (reported) |
| What the FE design identifies | Year FE absorb the global reporting trend **and** the global warming trend. The estimate therefore reflects responses to **year-to-year local temperature anomalies**, not long-run warming. This is the price of removing reporting bias, and the report states it plainly |
| Negative control (pass/fail) | The same model with geophysical counts as the outcome. Expected IRR = 1; equivalence bounds [0.90, 1.10] |
| Decision | IRR > 1 with 95% CI excluding 1, **and** the placebo IRR's 95% CI includes 1. If the placebo fails, H4 is reported as **not credible as specified** |
| SESOI | IRR 1.05 per degC |
| Secondary | Negative binomial with FE dummies; start years 1980 and 2000; events with >= 100 deaths only; adding a precipitation anomaly |
| Exploratory | Separate subgroups (floods vs droughts may respond in opposite directions) |

### 2.8 H5: Risk segmentation

| Field | Specification |
|-------|---------------|
| Unit | Country (cross-section); countries missing any feature are excluded, and the count is reported |
| Features (fixed now) | (1) Warming rate, Sen slope from H1c (hazard). (2) Climate-related disasters per million people per year, 2000-latest, log1p (hazard). (3) Agricultural share of GDP, mean of the last 10 years (exposure). (4) ND-GAIN vulnerability (latest). (5) ND-GAIN readiness (latest, adaptive capacity) |
| Preprocessing | Log transforms as stated; winsorize at the 1st and 99th percentiles; z-standardize. No PCA, to keep the features interpretable |
| Primary | k-means (50 initializations), k in {3,...,7}; choose the k with the highest mean silhouette; ties go to the smaller k |
| Stability | 500 bootstrap resamples of countries. Mean ARI against the full-sample solution; per-cluster Jaccard (Hennig 2007; > 0.75 = stable); share of countries that change segment (R2) |
| Decision | Silhouette > 0.25 and mean bootstrap ARI > 0.6 |
| Interpretation | Silhouette 0.25-0.50 = weak structure (Kaufman & Rousseeuw 1990), so segments are called "screening groups". Above 0.50, "distinct segments" |
| Secondary | Ward hierarchical clustering and a Gaussian mixture; agreement with k-means (ARI) |
| Note | Descriptive; no p-values |

### 2.9 Predictive tasks (evaluation pre-specified; Phase 8)

| Task | Target | Validation | Baselines | Metrics and criteria |
|------|--------|-----------|-----------|----------------------|
| Q2 forecast | Monthly GMST anomaly (GISTEMP), horizons 1-12 months | Rolling origin, monthly origins from 2017-01 | Persistence, climatology, linear trend, ARIMA | MAE, RMSE, MASE; 80% and 95% interval coverage |
| Q3 yield model | ln(yield) per country-year-crop | Temporal (train <= 2010, validation 2011-16, test >= 2017, **evaluated once**) and GroupKFold by country | Country mean, country linear trend, the H3 panel FE model | RMSE (guideline: >= 10% better than the best baseline); hot-season subset error (R3); 90% interval coverage (R4) |

## 3. Units of analysis and granularity

The authoritative, machine-readable contracts are in `config/abt_schemas.yaml`;
this section explains the rules behind them.

### 3.1 Analytical Base Tables

| ABT | Grain | Primary key | Coverage | Expected size | Used by |
|-----|-------|-------------|----------|---------------|---------|
| `abt_global_monthly` | Global x month | `date` | 1850-01 to 2025-12 | exactly 2,112 rows | H1a (secondary), H2, Q2 forecast |
| `abt_global_annual` | Global x year | `year` | 1850-2025 | exactly 176 rows | H1a, H1b |
| `abt_country_year` | Country x year (balanced grid) | `iso3, year` | 1961-2025 | 150-260 countries x 65 years | H1c, H4, Q5 inputs |
| `abt_crop_panel` | Country x harvest year x crop | `iso3, year, crop_code` | 1961-latest FAOSTAT year | unbalanced | H3, Q3 model |
| `abt_country_profile` | Country | `iso3` | cross-section | one row per country | H5, dashboard |

### 3.2 Conventions

1. **Keys.** Countries: ISO 3166-1 alpha-3. Crops: FAOSTAT item codes (DEC-011). Months: month-start dates.
   Regional and income aggregates (e.g. "World", "Africa") never appear as keys.
2. **Balanced grids.** `abt_country_year` holds every country x every year; missing values are NaN.
   This makes lags, completeness checks and missingness maps reliable.
3. **Flags, not deletions.** ABTs are never filtered for analysis reasons; sample membership is a
   boolean `in_sample_*` column, so every exclusion is counted and auditable.
4. **Temporal aggregation.** Temperature: mean of 12 months; precipitation: sum of 12 months;
   annual values are NaN unless all 12 months are valid. ONI (3-month running mean) is assigned to its
   centre month (DJF -> January). Disaster events are assigned to their start year.
5. **Harvest-year rule (H3).** Growing-season climate is assigned to the FAOSTAT year (= harvest year).
   Seasons may begin in the previous calendar year (e.g. Southern-Hemisphere summer crops);
   `gs_spans_two_years` records this.
6. **Structural zeros (H4).** Inside the H4 window, a country-year with no EM-DAT record has 0 events.
   Outside the window, or before a country existed, the value is NaN. This assumes EM-DAT's inclusion
   criteria are applied consistently; reporting gaps are handled by year FE and the geophysical
   placebo (section 2.7).
7. **Fixed classification.** World Bank region and income group come from one reference
   classification year (recorded in the manifest), so composition changes do not mix with real effects.
8. **Panel operations.** Lags and differences are computed within `iso3`, sorted by `year`; the first
   year of each country is NaN. Values never cross from one country to another.

### 3.3 Join rules

- Joins always use the full key; every merge declares its cardinality
  (`pandas.merge(..., validate="one_to_one" | "many_to_one")`), which raises on duplicate keys.
- Every left join asserts that the row count is unchanged (no fan-out).
- Crop rows join country-year rows on `(iso3, year)` (many-to-one); country-level quantities are
  never summed over crop rows.

### 3.4 Contract checks (enforced in Phase 4)

Unique primary keys · non-null keys · declared dtypes · values within plausibility ranges · exact row
counts for global tables · complete monthly/annual index without gaps · balanced country-year grid ·
foreign keys resolve · `iso3` values are valid ISO codes (no aggregates) · annual = mean of
monthly within 1e-9 · 1991-2020 mean of every rebased anomaly series = 0 within 1e-6.

### 3.5 Level of inference

All results are at the stated grain. Country-level associations are not claimed to hold for
sub-national regions, farms or individuals (ecological fallacy; see 1.4).

## 4. Study window

### 4.1 Windows by analysis

All windows are fixed by external criteria (data availability, WMO conventions, record start dates),
never by inspecting results. They are stored in `config/config.yaml` and checked by `validate_config`.

| Analysis | Window | Justification |
|----------|--------|---------------|
| H1a, H1c (formal trend tests) | 1961-2025 | FAOSTAT start; common to all datasets so results are comparable |
| H1b (structural break) | 1880-2025 | A 1970s break needs a long pre-break record (DEC-027) |
| Context charts | 1850/1880-2025 | Descriptive only; never used for tests |
| H2, Q2 forecast | 1959-01 to 2025-12 | First complete year of the Mauna Loa CO2 record |
| H3, Q3 model | 1961 to FAOSTAT effective end | FAOSTAT coverage |
| H4 | 1990-2025 primary; 1980 and 2000 sensitivity | Disaster-reporting completeness (DEC-030) |
| H5 features | Warming rate 1961-2025; disasters 2000-effective end; agricultural share last 10 years; ND-GAIN latest year | Recent conditions for risk; long record for the warming rate |
| Anomaly baseline | 1991-2020 | WMO climate normal (DEC-009) |
| Pre-industrial reference | 1850-1900 | IPCC AR6; context statements only |
| ML splits | Train <= 2010; validation 2011-2016; test 2017-effective end (>= 5 years) | Temporal hold-out; test set locked until Phase 9 |

The last complete year as of planning is 2025; partial 2026 data are excluded everywhere.

### 4.2 Effective-coverage policy (pre-registered contingencies)

Applying these rules in Phase 3 is part of the plan, not a deviation.

1. **Effective end year.** For each dataset, the last year that is complete:
   - monthly data: all 12 months present;
   - annual country data (FAOSTAT, WDI, EM-DAT): the number of reporting countries is >= 90% of the
     median count over the previous 5 years (this guards against a provisional, partly reported last year).
2. **Truncation, never extension.** Each analysis ends at the minimum of its planned end and the
   effective end of every dataset it uses. Missing final years are never imputed or extrapolated.
3. **Test-set contingency.** If truncation leaves fewer than `splits.min_test_years` (5) test years,
   the train/validation/test boundaries all shift earlier by the shortfall.
4. **Provisional values.** Values flagged as provisional or estimated stay in the data and are flagged;
   a robustness check excludes the last effective year (section 2.6, secondary analyses).
5. **"Latest" definitions.** ND-GAIN "latest" is the most recent year available at the access date.
   All "latest" years are recorded in the manifest and the data quality report.
6. **Vintage.** Coverage is assessed on the frozen download (DEC-024). New releases are not pulled in
   to extend windows during the analysis.

## 5. Risk and assumption register

Scale: Likelihood (L) and Impact (I) from 1 (low) to 3 (high); Priority P = L x I.
A **tripwire** is an observable signal that the risk is happening. When a tripwire fires:
(1) record it in the decision log, (2) apply the pre-registered response below,
(3) any response not listed here is a deviation (section 7).

### 5.1 Data risks

| ID | Risk | L | I | P | Tripwire (early-warning signal) | Pre-registered response | Checked in |
|----|------|---|---|---|---------------------------------|-------------------------|------------|
| RD1 | EM-DAT reporting improves over time, creating a false "more disasters" signal | 3 | 3 | 9 | Geophysical counts trend upward; placebo IRR CI excludes 1 | Year FE; placebo is pass/fail for H4; 1980/2000 starts; >= 100 deaths subset | 3, 6 |
| RD2 | Unit changes (FAOSTAT hg/ha vs kg/ha; ERA5 K and m/day) | 2 | 3 | 6 | Contract range violations; yield / (production/area) ratio of ~10 instead of ~1 | Explicit conversions with unit tests; one unit asserted per column after cleaning | 3, 4 |
| RD3 | Historical entities break series (USSR, Yugoslavia, Czechoslovakia, Sudan, Ethiopia/Eritrea, Serbia & Montenegro) | 3 | 2 | 6 | Unmatched names in the ISO3 crosswalk log; duplicate country-year keys | Crosswalk table; predecessor entities excluded from panels; successors enter when their data begin | 4 |
| RD4 | FAOSTAT values estimated or imputed, mostly in low-capacity countries | 3 | 2 | 6 | Share of flagged values > 30% for a country-crop | Keep flags; robustness excluding flagged values; R8 visual flags | 3, 6 |
| RD5 | ERA5 quality lower before the satellite era (pre-1979: fewer assimilated observations) | 2 | 2 | 4 | A step change in the ERA5 minus Berkeley country difference around 1979 | H1c requires agreement with Berkeley Earth; robustness on 1979-2025 | 3, 6 |
| RD6 | Sparse station coverage makes early country temperatures noisy | 2 | 2 | 4 | ERA5 vs Berkeley country correlation < 0.8 | Replication across both products; flag affected countries | 3 |
| RD7 | Simplified crop calendar (one season per country; ignores double-cropped rice and shifting planting dates) | 3 | 2 | 6 | Rice growing-season coefficients near zero, unlike the other crops | Document the limitation; gridded crop calendar as a stretch goal | 4, 6 |
| RD8 | Partial or provisional final year | 3 | 1 | 3 | Fewer reporting countries in the last year (DEC-037 rule) | Truncate to the effective end year; robustness excluding the last year | 3 |
| RD9 | Small countries and islands covered by few grid cells; coastal land-sea mixing | 3 | 1 | 3 | n_grid_cells < 10 | Flag (R8); robustness excluding flagged countries | 4 |
| RD10 | Data revised by providers during the project | 2 | 2 | 4 | `make verify-data` checksum mismatch | Frozen snapshot (DEC-024); deliberate refresh only, with a DEC entry | 2-11 |
| RD11 | EM-DAT export is manual, so not fully scriptable | 2 | 1 | 2 | — | Record export filters and date in the manifest | 2 |

### 5.2 Method and inference risks

| ID | Risk | L | I | P | Tripwire | Pre-registered response | Checked in |
|----|------|---|---|---|----------|-------------------------|------------|
| RM1 | Technology growth confounds the climate effect on yields | 3 | 3 | 9 | H3 results change sign or significance between trend specifications | Country quadratic trends (primary); linear trends and first differences as robustness | 6 |
| RM2 | Serial and spatial correlation make SEs too small | 3 | 3 | 9 | Residual autocorrelation; residual Moran's I across countries within a year | Two-way clustering (H3), HAC (H1/H2), block bootstrap (H1b) | 6 |
| RM3 | Few clusters for cluster-robust inference | 2 | 3 | 6 | Fewer than 40 countries in an estimation sample | Wild cluster bootstrap (section 2.0) | 6 |
| RM4 | Aggregation bias: country-season means smooth over damaging extremes (H3) | 3 | 2 | 6 | Lind-Mehlum fails while the linear temperature effect is negative | Report honestly as attenuation; degree-day features as a stretch goal | 6, 9 |
| RM5 | CO2 indistinguishable from any smooth trend (H2) | 3 | 2 | 6 | BIC difference < 2 between the ln(CO2) and time-trend models | State the identification limit explicitly | 6 |
| RM6 | H4 identifies only year-to-year responses (year FE absorb long-run warming) | 3 | 2 | 6 | Structural (always true) | Explicit language in every H4 statement | 6, 10 |
| RM7 | ML leakage (features from the future, scaling fitted on all data, lags crossing the split) | 2 | 3 | 6 | Test performance clearly better than validation performance | Phase 7 leakage audit; preprocessing fitted on training folds only | 7, 8 |
| RM8 | Forking paths and undisclosed flexibility | 2 | 3 | 6 | Any analysis choice not in this plan | Pre-registration; Deviations log; exploratory label | all |
| RM9 | Omitted volcanic and solar forcing (H2) | 2 | 2 | 4 | beta moves by more than 1 SE under the volcanic exclusion windows | Report the shift; volcanic aerosol regressor as a stretch goal | 6 |
| RM10 | No cointegration between GMST and ln(CO2) | 1 | 3 | 3 | Engle-Granger fails to reject | Pre-registered "inconclusive" path (section 2.5) | 6 |
| RM11 | Weak cluster structure (H5) | 3 | 1 | 3 | Silhouette < 0.5 | "Screening groups" wording (DEC-031) | 8 |

### 5.3 Engineering, communication and process risks

| ID | Risk | L | I | P | Tripwire | Pre-registered response | Checked in |
|----|------|---|---|---|----------|-------------------------|------------|
| RC1 | Over-claiming causality or attribution | 2 | 3 | 6 | Causal or attribution wording in drafts | R7 language rules; review checklist in Phase 10 | 10 |
| RC2 | Country rankings misused for stigmatization or as a sole funding criterion | 2 | 2 | 4 | Bare ranks without intervals | R1 rank intervals; R8 flags; screening-tool framing | 10 |
| RC3 | Benchmark mismatch misread (local growing-season degC vs global degC in Zhao et al. 2017) | 2 | 2 | 4 | Comparison stated without its caveat | Mandatory caveat next to every benchmark comparison | 10 |
| RE1 | ERA5 download size and CDS queue delays (~6.5 GB) | 2 | 2 | 4 | Requests queued for many hours | Year-by-year chunks with resume; fallback: World Bank CCKP aggregates, recorded as a deviation | 2 |
| RE2 | Dependency drift changes numerical results | 1 | 3 | 3 | Version-contract test failure | Pinned env + lock file + rebuild from file (DEC-019, DEC-022) | all |
| RP1 | No independent reviewer (solo project) | 3 | 2 | 6 | — | Self-review via the PR template; optionally ask a peer to review the plan before `prereg-v1` | 1, 10 |

### 5.4 Assumption register

| ID | Assumption | Needed by | Why plausible | How checked / if violated |
|----|------------|-----------|---------------|---------------------------|
| A1 | Conditional on fixed effects and trends, year-to-year weather is as good as random with respect to other yield and disaster drivers (Dell, Jones & Olken 2014) | H3, H4 | Weather is not chosen by farmers or governments | Placebo (H4); robustness to trend specification (H3) |
| A2 | Panel weather responses approximate climate responses; adaptation is limited over the horizon studied | H3 interpretation | Standard in the literature, but contested | Exploratory long-differences comparison (Burke & Emerick 2016); interpretation limited to short-run responses |
| A3 | Little spillover between countries (e.g. via world prices changing planting and inputs) | H3 | Year FE absorb global price shocks common to all countries | Year FE; acknowledged limitation |
| A4 | Errors in measuring climate are mostly random, which biases effects toward zero rather than away from it | H3, H4 | Reanalysis error is not driven by yields or disasters | Agreement of ERA5 and Berkeley Earth; the direction of any bias is stated |
| A5 | EM-DAT inclusion criteria are applied consistently across countries within a year | H4 | A single database with fixed thresholds | Placebo test; >= 100 deaths subset |
| A6 | Crop calendars are stable over the study period | H3 | Planting dates shift slowly | Limitation (RD7); stretch: time-varying gridded calendar |
| A7 | GMST and ln(CO2) are cointegrated | H2 | Physical forcing-response relationship | Engle-Granger test; "inconclusive" path if violated |
| A8 | The countries analysed represent the population of interest | H1c, H5 | All countries with data are included (census, not sample) | Missingness analysis vs GDP and region (Phase 3, Step 3.4); exclusions reported |
| A9 | Predecessor/successor border rules do not bias panels | H1c, H3, H4 | Rule fixed in advance (RD3) | Robustness excluding all countries affected by border changes |

## 6. Scope: MVP vs stretch

### 6.1 MVP (must ship)

Every confirmatory **and secondary** analysis in section 2 is MVP and is never cut for time.

| ID | Deliverable | Definition of done |
|----|-------------|--------------------|
| M1 | Q1 signal: H1a, H1b, H1c | All decision rules evaluated on all specified datasets; secondary analyses reported; country warming map with flags (R8) |
| M2 | Q2 drivers: H2 + 12-month forecast | Stationarity, cointegration and DOLS results; volcanic-exclusion robustness; forecast beats or is honestly compared with all baselines (section 2.9) |
| M3 | Q3 agriculture: H3 panel + yield ML model | Lind-Mehlum results for 4 crops; % per degC effects vs benchmark with caveat; ML model vs 3 baselines under temporal and country CV; R3 hot-season evaluation; R4 interval coverage |
| M4 | Q4 extremes: H4 | PPML IRR with CI; placebo pass/fail result; start-year and severity sensitivities |
| M5 | Q5 risk: H5 | Chosen k, silhouette, bootstrap ARI and Jaccard; R2 switching share; segment profiles |
| M6 | Dashboard (D5) | Deployed publicly; uncertainty shown for every country value (R1); data-quality flags (R8) |
| M7 | Reports (D2, D3, D6, D7) | Data quality report; statistical report; executive summary; data card; model card |
| M8 | Reproducibility (D1) | `make all` rebuilds every result from raw data; CI passes |

### 6.2 Stretch goals (ranked; dropped from the bottom up)

| Rank | ID | Stretch goal | Serves | Cost |
|------|----|--------------|--------|------|
| 1 | S1 | Long-differences estimator (exploratory) | H3 interpretation (A2) | Low |
| 2 | S2 | Cropland-weighted climate aggregation | H3, Q3 | Medium (cropland mask) |
| 3 | S3 | Volcanic aerosol regressor | H2 (RM9) | Medium (new dataset) |
| 4 | S4 | Degree-day features from daily ERA5 | H3, Q3 (RM4) | High (daily data, ~30x volume) |
| 5 | S5 | Gridded, time-varying crop calendar | H3 (RD7, A6) | Medium |
| 6 | S6 | Extreme value analysis (return-period shifts, GHCN-Daily) | Q1 extension | High |
| 7 | S7 | SPEI drought-index features | Q3, Q4 | Medium |
| 8 | S8 | Docker image / DVC pipeline | D1 | Medium |
| 9 | S9 | Sea-level and sea-ice indicator module | Q1 extension | Medium |

Ranking principle: stretch goals that address a named risk or assumption in section 5 rank above
those that add new topics.

### 6.3 Descoping and scope-change rules

1. **Order.** Stretch goals are dropped from rank 9 upward. Confirmatory and secondary analyses
   and requirements R1-R8 are never dropped for time.
2. **Trigger.** If a phase exceeds its planned time by more than 50%, the next stretch goal in the
   drop order is removed and the removal is logged (decision log + `docs/time_log.md`).
3. **Pre-registered fallbacks.** If ERA5 cannot be acquired within Phase 2's budget, country
   climate comes from World Bank CCKP aggregates (framework Step 2.4). Any other unavailable MVP
   input is a deviation (section 7), reported with its consequence.
4. **No silent additions.** Analyses added after `prereg-v1` are exploratory by definition, are
   labelled as such, and are started only after the MVP items they relate to are done.

### 6.4 Time budget (part-time, framework Appendix B)

| Weeks | Phases | Milestone |
|-------|--------|-----------|
| 1 | 0-1 | Repo live; plan frozen (`v0.0-setup`, `prereg-v1`) |
| 2 | 2 | Core data downloaded with manifest |
| 3 | 3 | Data quality report; effective coverage fixed (DEC-037) |
| 4-5 | 4 | ABTs validated against `abt_schemas.yaml` |
| 6 | 5 | EDA insight log |
| 7-8 | 6 | Statistical results register + robustness tables |
| 8-9 | 7-8 | Feature tables versioned; models tracked in MLflow |
| 10 | 9 | Locked test evaluation; interpretation; model card |
| 11 | 10 | Report + dashboard |
| 12 | 11-12 | CI, deployment, README, `v1.0` |

Actual time per phase is logged in `docs/time_log.md` (input to the Phase 12 retrospective).

## 7. Deviations log
| ID | Date | Section | Change | Why | Results seen before change? |
|----|------|---------|--------|-----|-----------------------------|
| DV1 | 2026-11-13 | 2.2, 2.3 (Berkeley global) | Berkeley's global file contains two series (air temperature over sea ice; water temperature under sea ice). Primary: the air-temperature-over-sea-ice series, which Berkeley Earth presents as its primary version; the Phase 4 parser must confirm each table's label from the file header. The water-temperature series is an exploratory sensitivity | The plan did not specify which series | No (only file structure inspected) |
| DV2 | 2026-11-13 | 2.4 (H1c replication) | Berkeley country replication covers countries whose newest Berkeley file spans 1961-2020 (195 files); 33 territories whose files end 2016-05 are outside the replication (flagged, not deleted) | Berkeley publishes two file generations; DEC-037 truncation | No |
