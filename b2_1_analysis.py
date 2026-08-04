"""
B2.1 — OLS Analysis of ERCOT Residual Demand and Spot Prices
Author: Suraj Togaru
ASSIP 2026, George Mason University — Department of Finance

Specification:
    spot_price_t = a + b*residual_demand_t + sum(h_i*hour_i) + sum(m_j*month_j) + e_t

Key result: b = $0.00083/MWh per MW  (t = 14.19),  R2 = 0.1659
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------
# Expected columns: timestamp, system_load_mw, wind_mw, solar_mw, hb_west_price
# Period: January 2022 - December 2023  (17,483 hourly obs.)

def load_ercot_panel(filepath: str) -> pd.DataFrame:
    """Load and validate the hourly ERCOT panel."""
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df):,} observations")
    print(f"Period: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    return df


# -----------------------------------------------------------
# 2. CONSTRUCT VARIABLES
# -----------------------------------------------------------

def build_features(df: pd.DataFrame):
    """
    Construct residual demand and fixed-effect dummies.

    Residual demand = system_load_mw - wind_mw - solar_mw
    When residual demand rises, scarcity pressure increases and
    spot prices spike, creating the price incentive to curtail.
    """
    df = df.copy()

    # Key independent variable
    df["residual_demand"] = df["system_load_mw"] - df["wind_mw"] - df["solar_mw"]

    # Fixed-effect dummies (hour 0 and January omitted as reference)
    df["hour"]  = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month

    hour_dummies  = pd.get_dummies(df["hour"],  prefix="hour",  drop_first=True)
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True)

    df = pd.concat([df, hour_dummies, month_dummies], axis=1)
    return df, list(hour_dummies.columns), list(month_dummies.columns)


# -----------------------------------------------------------
# 3. OLS WITH HC1-ROBUST STANDARD ERRORS
# -----------------------------------------------------------

def run_ols(df, hour_cols, month_cols):
    """
    Estimate OLS: spot_price ~ residual_demand + hour FE + month FE
    Standard errors: HC1-robust (heteroskedasticity-consistent).
    """
    X_cols = ["residual_demand"] + hour_cols + month_cols
    X = sm.add_constant(df[X_cols])
    y = df["hb_west_price"]
    model = sm.OLS(y, X).fit()
    return model.get_robustcov_results(ctype="HC1")


def print_results(results) -> None:
    """Print key regression output."""
    print("=" * 60)
    print("B2.1 OLS RESULTS — HC1-Robust Standard Errors")
    print("=" * 60)

    beta  = results.params["residual_demand"]
    t_val = results.tvalues["residual_demand"]
    p_val = results.pvalues["residual_demand"]

    print("\nResidual demand (beta): $%.5f/MWh per MW" % beta)
    print("t-statistic:            %.2f" % t_val)
    print("p-value:                %.4f" % p_val)
    print("R-squared:              %.4f" % results.rsquared)
    print("N:                      %s"   % f"{int(results.nobs):,}")

    print("\n--- Hour-of-Day Fixed Effects (top 5 by magnitude) ---")
    hour_fe = {k: v for k, v in results.params.items() if k.startswith("hour_")}
    for h, coef in sorted(hour_fe.items(), key=lambda x: -x[1])[:5]:
        print("  %s: +$%.2f" % (h, coef))

    print("\n--- Month Fixed Effects ---")
    month_names = {2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    month_fe = {k: v for k, v in results.params.items() if k.startswith("month_")}
    for m_col, coef in sorted(month_fe.items(), key=lambda x: -x[1]):
        m_num = int(m_col.split("_")[1])
        sign  = "+" if coef >= 0 else ""
        print("  %s: %s$%.2f" % (month_names.get(m_num, m_col), sign, coef))


# -----------------------------------------------------------
# 4. ROBUSTNESS — SCARCITY THRESHOLDS
# -----------------------------------------------------------

SCARCITY_THRESHOLDS = [80, 122, 160, 200]   # $/MWh

def scarcity_robustness(df, hour_cols, month_cols) -> pd.DataFrame:
    """
    Re-estimate the residual demand coefficient restricting to
    hours where price >= threshold.  The S19/S21 break-even is
    ~$122/MWh; checking $80-$200 validates stability.
    """
    records = []
    X_cols  = ["residual_demand"] + hour_cols + month_cols
    for thresh in SCARCITY_THRESHOLDS:
        subset = df[df["hb_west_price"] >= thresh]
        if len(subset) < 100:
            continue
        X   = sm.add_constant(subset[X_cols])
        y   = subset["hb_west_price"]
        res = sm.OLS(y, X).fit().get_robustcov_results(ctype="HC1")
        records.append(dict(
            threshold = thresh,
            n_obs     = len(subset),
            beta      = res.params["residual_demand"],
            t_stat    = res.tvalues["residual_demand"],
            r2        = res.rsquared,
        ))
    rob = pd.DataFrame(records)
    print("\n--- Robustness: beta at each scarcity threshold ---")
    print(rob.to_string(index=False))
    return rob


# -----------------------------------------------------------
# 5. CURTAILMENT IMPLICATIONS
# -----------------------------------------------------------

BREAK_EVEN     = 122    # $/MWh  — S19/S21-class ASIC
MINING_LOAD_MW = 100    # stylized facility size
EMISSIONS_LOW  = 0.25   # tCO2/MWh
EMISSIONS_HIGH = 0.70   # tCO2/MWh

def curtailment_implications(df) -> None:
    """
    Implied curtailment and avoided emissions for a stylized
    100 MW Bitcoin mining facility at the ~$122/MWh break-even.
    """
    scarcity   = df[df["hb_west_price"] >= BREAK_EVEN]
    n_hours    = len(scarcity)
    mwh        = n_hours * MINING_LOAD_MW

    print("\n" + "=" * 60)
    print("CURTAILMENT IMPLICATIONS — 100 MW Mining Load")
    print("=" * 60)
    print("Break-even:            ~$%d/MWh (S19-class ASIC)" % BREAK_EVEN)
    print("Scarcity hours:         %s" % f"{n_hours:,}")
    print("Annual curtailment:     %s MWh" % f"{mwh:,}")
    print("Avoided emissions low:  %s tCO2" % f"{mwh * EMISSIONS_LOW:,.0f}")
    print("Avoided emissions high: %s tCO2" % f"{mwh * EMISSIONS_HIGH:,.0f}")


# -----------------------------------------------------------
# 6. VISUALIZATIONS
# -----------------------------------------------------------

def plot_results(df, results, save_path=None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("B2.1 — ERCOT Residual Demand & Spot Prices", fontsize=14)

    # A: scatter
    ax = axes[0, 0]
    ax.scatter(df["residual_demand"] / 1000, df["hb_west_price"],
               alpha=0.1, s=5, color="steelblue")
    ax.set_xlabel("Residual Demand (GW)")
    ax.set_ylabel("HB_WEST Spot Price ($/MWh)")
    ax.set_title("A: Residual Demand vs. Spot Price")
    ax.set_ylim(-50, 500)

    # B: hour fixed effects
    ax = axes[0, 1]
    hour_fe = {int(k.split("_")[1]): v for k, v in results.params.items()
               if k.startswith("hour_")}
    hours = sorted(hour_fe.keys())
    ax.bar(hours, [hour_fe[h] for h in hours], color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Price Premium vs. Midnight ($/MWh)")
    ax.set_title("B: Hour-of-Day Fixed Effects")

    # C: monthly median price
    ax = axes[1, 0]
    month_avg    = df.groupby("month")["hb_west_price"].median()
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.bar(range(1, 13), [month_avg.get(m, 0) for m in range(1, 13)], color="coral")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels, rotation=45)
    ax.set_ylabel("Median Spot Price ($/MWh)")
    ax.set_title("C: Median Price by Month")

    # D: residual demand time series
    ax = axes[1, 1]
    weekly = df.set_index("timestamp")["residual_demand"].resample("W").mean() / 1000
    ax.plot(weekly, color="steelblue", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual Demand (GW)")
    ax.set_title("D: Weekly Avg Residual Demand")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

if __name__ == "__main__":
    DATA_PATH = "data/ercot_hourly_2022_2023.csv"

    df_raw = load_ercot_panel(DATA_PATH)
    df, hour_cols, month_cols = build_features(df_raw)

    results = run_ols(df, hour_cols, month_cols)
    print_results(results)

    scarcity_robustness(df, hour_cols, month_cols)
    curtailment_implications(df)
    plot_results(df, results, save_path="b2_1_results.png")
