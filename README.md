# B2.1 — Price-Responsive Dispatch in ERCOT: OLS Analysis of Residual Demand and Spot Prices

**Author:** Suraj Togaru
**Sub-project:** B2.1 (Price Signal & Dispatch Mechanism)
**Group:** B2 — Load Flexibility, AI Workloads, and Market Structure
**Program:** ASSIP 2026, George Mason University — Department of Finance

---

## Research Question

Does ERCOT's residual demand signal reliably trigger price-responsive dispatch from large flexible computing loads (Bitcoin mining farms and AI data centers)? And what actually constrains their flexibility?

---

## Data

| Variable | Description |
|---|---|
| **Source** | ERCOT hourly settlement point prices + load/generation reports |
| **Period** | January 2022 – December 2023 |
| **Observations** | 17,483 hourly obs. |
| **Price** | HB_WEST real-time spot price ($/MWh) |
| **Load** | System-wide load (MW) |
| **Renewables** | Total wind + solar generation (MW) |
| **Residual demand** | System load minus total renewable generation (MW) |

**Break-even threshold:** ~$122/MWh for S19/S21-class Bitcoin mining ASICs (derived from hardware efficiency specs and network difficulty estimates).

---

## Methods

### OLS Regression Specification

```
spot_price_t = α + β · residual_demand_t + Σ γ_h · hour_h + Σ δ_m · month_m + ε_t
```

- **Dependent variable:** HB_WEST real-time spot price ($/MWh)
- **Key independent variable:** Residual demand (MW) = system load − (wind + solar)
- **Controls:** Hour-of-day fixed effects (23 dummies, hour 0 omitted), month fixed effects (11 dummies, January omitted)
- **Standard errors:** HC1-robust (heteroskedasticity-consistent)
- **Estimator:** OLS via statsmodels with get_robustcov_results(ctype='HC1')
- **Robustness:** Coefficient tested at scarcity thresholds of $80, $122, $160, and $200/MWh

### Residual Demand Definition

Residual demand captures the net load served by dispatchable (non-renewable) generation. When residual demand rises, spot prices spike — creating the price incentive for flexible computing loads to curtail.

---

## Results

### Main Regression

| Parameter | Estimate | t-statistic |
|---|---|---|
| Residual demand (β) | **$0.00083/MWh per MW** | **14.19** |
| Hour 20 fixed effect | +$54.47 | — |
| August fixed effect | +$47.88 | — |
| R² | 0.1659 | — |
| N | 17,483 | — |

The coefficient is statistically significant (t = 14.19) and stable across robustness checks at $80–$200/MWh scarcity thresholds.

### Selected Fixed Effects

**Hour-of-day premiums** (relative to Hour 0):
- Hour 17: +$18.20
- Hour 18: +$28.43
- **Hour 20: +$54.47** ← evening peak
- Hour 21: +$46.11

**Month premiums** (relative to January):
- June: +$22.14
- July: +$38.65
- **August: +$47.88** ← heat-wave premium
- September: +$19.33

### Implied Curtailment — Stylized 100 MW Mining Load

At the ~$122/MWh S19-class break-even, a 100 MW mining facility curtails during ~**1,005 scarcity hours/year**:

| Metric | Value |
|---|---|
| Annual curtailment | **100,500 MWh** |
| Avoided emissions — mining (low) | 25,125 tCO2 |
| Avoided emissions — mining (high) | 70,350 tCO2 |
| Avoided emissions — AI training (low) | 6,281 tCO2 |
| Avoided emissions — AI training (high) | 24,623 tCO2 |

*Ranges reflect marginal emissions rate sensitivity: 0.25–0.70 tCO2/MWh*

---

## Key Finding: Workload Economics Wall, Not a Hardware Wall

The flexibility gap between Bitcoin mining and AI data centers is **not about hardware response speed** — it is about workload economics:

- **Mining ASICs (κ ≈ 0):** Near-zero switching cost. Firmware sleep mode activates in seconds. Miners curtail whenever spot price exceeds the ~$122/MWh break-even.
- **AI inference / SLA-bound (κ high):** Contractually fixed regardless of technical capability. Latency SLAs make curtailment infeasible even when hardware could respond instantly.
- **AI training (κ > 0, small):** Moderately flexible — batch jobs shift hours without losing work, but coordination overhead limits speed.

**Conclusion:** Workload classification — not warm-up time — is the binding constraint on computing-load flexibility in ERCOT. The residual demand coefficient validates the price-responsiveness conditions assumed in demand-response theory.

---

## Poster Copy-Paste Content

### Results Section (paste under "B2.1:" in poster)

- Residual demand significantly predicts ERCOT spot prices (coeff = $0.00083/MWh per MW, t = 14.19), with evening peak (Hour 20: +$54.47) and August heat-wave (+$47.88) fixed-effect premiums.
- The $0.00083/MWh coefficient and ~$122/MWh S19-class break-even imply miners curtail during roughly 1,005 annual scarcity hours, validating the price-incentive side of the d(·) dispatch rule.
- Model explains 16.6% of hourly price variation; residual demand is the primary structural driver of price-responsive miner behavior in ERCOT.

### Methods Section (paste under "B2.1 Methods:" header in poster)

- Constructed hourly ERCOT panel (Jan 2022–Dec 2023; 17,483 obs.) linking system load, wind, solar, and HB_WEST spot prices.
- Defined residual demand as system load minus total renewable generation; identified ~$122/MWh as the S19-class mining break-even.
- Estimated OLS: spot price ~ residual demand + hour-of-day + month fixed effects with HC1-robust standard errors.

---

## Dependencies

```
pandas
numpy
statsmodels
matplotlib
```

`pip install pandas numpy statsmodels matplotlib`
