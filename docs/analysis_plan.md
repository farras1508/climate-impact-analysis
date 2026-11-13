# Analysis Plan (Pre-registration)

**Status:** DRAFT — not yet frozen. Will be tagged `prereg-v1` at the end of Phase 1.
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
| Exploratory | A possible post-2010 acceleration. The post-break segment is too short for confirmatory inference |

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
_Step 1.4 — to be written._

## 5. Risk and assumption register
_Step 1.5 — to be written._

## 6. Scope: MVP vs stretch
_Step 1.6 — to be written._

## 7. Deviations log
_Empty until the plan is frozen. Format: date · section changed · what · why · results seen before the change? (yes/no)._
