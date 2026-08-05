# ERCOT Grid-Side Evidence for Flexible-Load Response
### Institutional Mechanisms, Data Construction, Large-Load Context, and Interpretation Limits

**Author:** Shreyas Kalidindi  
**Sub-project:** B2.2 — ERCOT Grid-Side Evidence  
**Group:** B2 — Load Flexibility & AI Workloads  
**Program:** ASSIP 2026, George Mason University — Department of Finance

---

This note combines the four short-note deliverables produced for B2.2 into a single narrative: the ERCOT mechanisms that give large, flexible loads a reason to change consumption; the construction of the hourly residual-demand and price panel used to test that behavior; the broader context of large-load capacity growth in ERCOT over the same period; and, finally, a precise statement of what the resulting evidence does and does not show.

---

## Repository Contents

```
B2.2/
├── README.md                        ← this file
├── B2_2_ERCOT_Grid_Evidence.py      ← analysis script (run this)
└── outputs/
    ├── three_event_comparison.png
    ├── event_summary_table.csv
    └── event_window_data.csv
```

**To run:** Place `ercot_master_panel.csv` in the same folder as the script, then run:
```bash
pip install pandas numpy matplotlib
python B2_2_ERCOT_Grid_Evidence.py
```

---

## 1. Controllable Load Resources (CLR) and Four Coincident Peak (4CP) in ERCOT

### Controllable Load Resources (CLR)

A Controllable Load Resource (CLR) is a large electricity consumer that agrees to cut its electricity use when asked by ERCOT. Instead of generating additional electricity during peak demand, ERCOT can reduce system demand by dispatching qualified load resources. From the grid's perspective, less demand is the same as more electricity generation. CLRs have the ability to operate in real-time within ERCOT to support the reliability of the system (ERCOT, 2024). One example of a flexible load that may qualify for this type of participation is a Bitcoin mining facility. Mining computers can generally be turned off and turned back on quickly without harming the equipment or needing long recovery times.

During a period of high electricity prices or low supply, operators can temporarily decrease their power consumption and return to normal operation when the grid situation improves. This flexibility differs from many traditional industrial processes that require longer shutdown and restart procedures. CLR participation is relevant to this project as it provides a real-world mechanism for flexible loads that respond to scarcity. As residual demand and electricity prices increase, the incentive to reduce consumption for qualifying loads increases. The event-study analysis tests whether this behavior is present in the ERCOT data.

### Four Coincident Peak (4CP)

The Four Coincident Peak (4CP) framework has been used to allocate transmission costs in ERCOT. ERCOT annually identifies the highest system demand interval for each of the months June, July, August, and September. Transmission charges for eligible large customers were linked to their electricity demand during these coincident peak intervals (ERCOT, 2024).

Many industrial customers tracked grid conditions through the summer and cut back on demand voluntarily if they anticipated a coincident peak, because of the potential for high transmission charges. This strategy could reduce their transmission charges while also lowering demand during periods of high system load.

ERCOT has made changes to its transmission cost allocation process over the years, but the economic concept behind 4CP still stands. Large electricity consumers respond to financial incentives to shift or reduce demand during periods of high system load — behavior very similar to the demand response mechanism discussed in this work.

### Relevance to This Study

CLRs are a formal market tool that allows large consumers of electricity to reduce their demand when requested by ERCOT. The historical 4CP program is another example of economic incentives causing large customers to shift electricity consumption away from times of peak demand. The hourly residual-demand series and ERCOT price data developed in this project can be used to empirically test these ideas. The analysis compares residual demand and electricity prices during periods of grid stress to identify the conditions under which flexible loads may have incentives to reduce electricity consumption as scarcity increases.

---

## 2. Data Construction

The data used for analysis comprises an hourly ERCOT series from January 1, 2022 to December 31, 2023. The hourly data contains total electricity demand, solar generation, and wind generation. Renewable energy generation is computed as the sum of solar and wind generation for each hour, and residual demand is defined as:

```
Residual Demand = Demand − (Solar + Wind)
```

Residual demand represents the remaining electricity demand after observed wind and solar generation are subtracted. ERCOT data is then combined with HB_WEST price data. The price data include the number of 15-minute periods in an hour, the hourly average price, the maximum 15-minute price in an hour, and two scarcity indicators. `scarcity_flag_any` equals 1 whenever at least one 15-minute period exceeds $122/MWh within an hour.

There is one difference in the way that the two datasets use hours. While the ERCOT dataset uses hour-beginning times from 0:00 to 23:00, HB_WEST uses hour-endings from 1 through 24. This difference was accounted for during alignment: when the ERCOT observation hour was h, the corresponding HB_WEST observation was the h+1 hour on the same day. The merged data included all 17,520 observations of the ERCOT dataset. There were no corresponding HB_WEST observations for 37 hours, so the associated price and scarcity variables were left missing.

Three periods of ERCOT grid stress were selected for the event-study analysis:

| Event | Event Center (t=0) | Window | Obs. | Missing HB_WEST Hrs. |
|---|---|---|---|---|
| July 2022 heat wave | Hour of highest `max_price_mwh` in candidate period | ±72 hrs | 145 | 5 |
| Winter Storm Elliott (Dec 2022) | Hour of highest `max_price_mwh` in candidate period | ±72 hrs | 145 | 2 |
| August 2023 heat wave | Hour of highest `max_price_mwh` in candidate period | ±72 hrs | 145 | 2 |

A ±72-hour window was extracted around each event center, yielding 145 hourly observations per event. Hours with missing HB_WEST prices were retained as missing rather than treated as zero-price or non-scarcity hours. Some HB_WEST observations were based on fewer than four 15-minute intervals and were retained as reported in the source data.

Aligning all three events on a common event-time clock — hours relative to t = 0 — makes them directly comparable rather than comparing across unrelated calendar dates.

---

## 3. Key Figure

![Three Event Comparison](outputs/three_event_comparison.png)

*Figure 1. Residual demand, HB_WEST price, and scarcity flags across three ERCOT grid-stress events (±72 hours around each event center), 2022–2023. Values plotted as-is; no smoothing, interpolation, or filling. Amber bands = scarcity hours. Gray bands = missing price data. Dashed line = t = 0.*

---

## 4. Large Load Capacity Growth in ERCOT

### Definition and Background

ERCOT defines a Large Load as one or more facilities at a single site with aggregate peak demand of at least 75 MW. In March 2022, ERCOT introduced an interim Large Load Interconnection Process, which had previously been handled separately by Transmission Service Providers. This process was later approved through PGRR115, effective December 15, 2025.

Terminology differs across sources. EIA used the term "large flexible load" (LFL) in its 2024 reporting, while current ERCOT materials use Large Load and separately identify large electronic loads, including data centers and cryptominers. ERCOT terminology is used here except when reporting EIA figures.

In its September 2024 Short-Term Energy Outlook, EIA reported 5,479 MW of approved LFL capacity, including 1,570 MW approved during the previous twelve months. EIA's 9,500 MW end-2025 figure was a modeling assumption rather than an observed value. The related estimate of 54 billion kWh of LFL consumption in 2025 was also a forecast based on assumed capacity and an assumed utilization rate of about 65% — not an observed outcome.

### Approved Capacity and Observed Load

ERCOT's March 13, 2026 Large Load Interconnection Status Update reported 9,042 MW with Approval to Energize. ERCOT distinguishes this from projects classified as Observed Energized and from approved projects not observed to be operational. A November 18, 2025 growth chart identified 5,302 MW as Observed Energized.

The March 13, 2026 update also reported a simultaneous monthly peak of 3,801 MW and a non-simultaneous monthly peak of 3,883 MW, with both figures labeled March 2025 in the presentation. These differences should not be interpreted as a utilization rate because the measures reflect different project statuses and reporting periods. Aggregate figures alone cannot identify curtailment.

### Relevance to the Three-Event Study

The July 2022, December 2022, and August 2023 event windows occurred during a broader period of growth in large loads within ERCOT. However, published aggregate figures do not show how much flexible load was approved, energized, or actively consuming electricity during each event. Differences across the three events therefore cannot be attributed to changes in the large-load population without more detailed load-level data.

### Limitations

The 9,500 MW and 54 billion kWh figures are an EIA modeling assumption and forecast respectively, not observed outcomes. The March 13, 2026 ERCOT update labels its peak-consumption figures as March 2025, so those values are reported as stated rather than reinterpreted. Aggregate large-load figures include data centers and industrial facilities as well as cryptocurrency mining and cannot be attributed specifically to Bitcoin mining. No figure in this section directly measures curtailment.

---

## 5. Claim Mapping and Interpretation Limits

**L5 (Flexible-load response to surplus and scarcity):** Figure 1 compares residual demand, prices, and scarcity across the three events, showing the grid-side conditions to which flexible loads may respond. However, it does not directly test whether miners curtailed because no miner-level or large-load consumption series is included.

**L4 (Real-time miner switching):** Figure 1 shows the price and scarcity signals that may lead flexible loads to reduce consumption but does not show the corresponding load response. It therefore cannot determine whether or how quickly miners adjusted consumption.

**L1 (Rapid scalability of mining load):** Figure 1 does not test rapid ramping because it uses hourly grid-level data rather than direct engineering or load-response measurements.

### Interpretation Limits

The original work plan included Winter Storm Heather (January 2024), but the assembled panel ends in December 2023. Three in-sample events were therefore used instead. This analysis should not be used to measure Bitcoin-miner consumption, demonstrate curtailment, or assume that flexible loads caused observed changes in prices or demand. It characterizes grid conditions only.

---

## References

ERCOT. (2024). *Load Resources in the ERCOT Market.* Electric Reliability Council of Texas. https://www.ercot.com/services/programs/load

ERCOT. (2024). *Four Coincident Peak (4CP) Information.* Electric Reliability Council of Texas. https://www.ercot.com/mktinfo/data_agg/4cp

U.S. Energy Information Administration. (2025). *Today in Energy: Large Flexible Loads in ERCOT.* https://www.eia.gov/todayinenergy/detail.php?id=63344
