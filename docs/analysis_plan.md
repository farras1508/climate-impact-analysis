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

---

## 2. Hypothesis register
_Step 1.2 — to be written._

## 3. Units of analysis and granularity
_Step 1.3 — to be written._

## 4. Study window
_Step 1.4 — to be written._

## 5. Risk and assumption register
_Step 1.5 — to be written._

## 6. Scope: MVP vs stretch
_Step 1.6 — to be written._

## 7. Deviations log
_Empty until the plan is frozen. Format: date · section changed · what · why · results seen before the change? (yes/no)._
