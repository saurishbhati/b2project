"""
B2.2 — ERCOT Grid-Side Evidence: Three-Event Analysis of Residual Demand, Prices, and Scarcity
Author: Shreyas Kalidindi
Sub-project: B2.2 (ERCOT Grid-Side Evidence)
Group: B2 — Load Flexibility, AI Workloads, and Market Structure
Program: ASSIP 2026, George Mason University — Department of Finance

=============================================================
PROJECT MOTIVATION
=============================================================
Large computing facilities are one of the fastest-growing sources
of electricity demand, raising important questions about how these
loads interact with the modern power system. Bitcoin mining has
emerged as a highly flexible computing load, while AI and cloud
workloads are often assumed to be less responsive. This project
characterizes ERCOT grid conditions surrounding major stress events
to evaluate whether flexible loads face economic incentives to curtail.

=============================================================
RESEARCH QUESTION
=============================================================
How do ERCOT grid conditions evolve during periods of electricity
market stress, and do these conditions create the market incentives
required for flexible computing loads to curtail consumption?

=============================================================
DATA
=============================================================
Source   : EIA-930 (demand, wind, solar) + ERCOT HB_WEST prices
Period   : January 2022 - December 2023 (17,520 hourly observations)
File     : ercot_master_panel.csv

Variables:
  demand_mw          - ERCOT system demand (MW)
  wind_mw            - Wind generation (MW)
  solar_mw           - Solar generation (MW)
  renewable_gen_mw   - Wind + solar (MW)
  residual_demand_mw - System demand minus renewables (MW)
  avg_price_mwh      - HB_WEST avg price per hour ($/MWh)
  max_price_mwh      - HB_WEST max price per hour ($/MWh)
  scarcity_flag_any  - 1 if any scarcity in that hour, else 0

Data quality notes:
  - Numeric columns stored with comma separators; parsed to float here
  - 37 rows have missing price/scarcity data; retained as gaps, not filled
  - 2 spring-forward gaps; 2 fall-back duplicates (none in event windows)
  - August 2023 event center built from 2 of 4 sub-hourly intervals

=============================================================
METHODS
=============================================================
1. Load and parse the hourly ERCOT master panel
2. For each of three candidate periods, identify the single hour
   with the highest max_price_mwh as the event center (t=0)
3. Extract a symmetric +/-72 hour window around each event center
4. Report data quality for each window
5. Produce a three-panel event-study figure and two summary CSVs

Three events:
  - 2022 Summer Heat     : candidate Jul 11-20 2022
  - Winter Storm Elliott : candidate Dec 22-25 2022
  - August 2023 Heat     : candidate Aug 1-31 2023

Note: Winter Storm Heather (Jan 2024) was the original target for
L4/L5 testing but falls outside the 2022-2023 panel window.
August 2023 was substituted as an in-sample alternative.

=============================================================
KEY FINDINGS
=============================================================
Event             Scarcity Hrs  Peak Price  Avg Resid Demand
2022 Summer Heat      29        $991/MWh     51,925 MW
Winter Storm Elliott  13        $924/MWh     42,954 MW
August 2023 Heat      29        $993/MWh     50,915 MW

Residual demand, HB_WEST prices, and scarcity co-move across all
three events, consistent with the demand-response mechanism the
model predicts. This analysis characterizes grid conditions only
and does NOT measure miner curtailment directly.

=============================================================
OUTPUTS
=============================================================
  three_event_comparison.png  - three-panel event-study figure
  event_summary_table.csv     - one row per event
  event_window_data.csv       - all 435 window rows (3 x 145)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os

# -----------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------

DATA_PATH  = "ercot_master_panel.csv"   # put this file in the same folder
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EVENTS = {
    '2022 summer heat':     ('2022-07-11', '2022-07-20'),
    'Winter Storm Elliott': ('2022-12-22', '2022-12-25'),
    'August 2023 heat':     ('2023-08-01', '2023-08-31'),
}

# -----------------------------------------------------------
# 1. LOAD
# -----------------------------------------------------------

def load_panel(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df['ts'] = pd.to_datetime(df['local_time'], format='%m/%d/%Y %H:%M', errors='coerce')

    def parse(s):
        return pd.to_numeric(
            s.str.strip().str.replace(',', '', regex=False).replace('', np.nan),
            errors='coerce')

    for c in ['demand_mw','solar_mw','wind_mw','renewable_gen_mw',
              'residual_demand_mw','avg_price_mwh','max_price_mwh',
              'scarcity_flag_any','n_intervals']:
        df[c + '_num'] = parse(df[c])

    print(f"Loaded {len(df):,} rows  |  {df['ts'].min().date()} to {df['ts'].max().date()}")
    return df

# -----------------------------------------------------------
# 2. SELECT EVENT CENTERS
# -----------------------------------------------------------

def select_centers(df):
    centers = {}
    for name, (a, b) in EVENTS.items():
        sub = df[(df['ts'] >= pd.Timestamp(a)) & (df['ts'] <= pd.Timestamp(b + ' 23:00'))]
        mx  = sub['max_price_mwh_num'].max()
        tied = sub[sub['max_price_mwh_num'] == mx]
        if len(tied) > 1:
            raise ValueError(f"Tie at max price for {name}: {tied['ts'].tolist()}")
        t0 = tied['ts'].iloc[0]
        print(f"{name}: t=0 = {t0}  max_price = {mx}  n_intervals = {tied['n_intervals'].iloc[0]}")
        centers[name] = t0
    return centers

# -----------------------------------------------------------
# 3. EXTRACT WINDOWS
# -----------------------------------------------------------

ORIG = ['local_time','demand_mw','solar_mw','wind_mw','renewable_gen_mw',
        'residual_demand_mw','n_intervals','avg_price_mwh','max_price_mwh',
        'scarcity_flag_any']
NUMS = ['residual_demand_mw_num','avg_price_mwh_num','max_price_mwh_num',
        'scarcity_flag_any_num','n_intervals_num']

def extract_windows(df, centers):
    frames = []
    for name, c0 in centers.items():
        lo = c0 - pd.Timedelta(hours=72)
        hi = c0 + pd.Timedelta(hours=72)
        w  = df[(df['ts'] >= lo) & (df['ts'] <= hi)].sort_values('ts').copy()
        missing = [t for t in pd.date_range(lo, hi, freq='h') if t not in set(w['ts'])]
        empty_p = w[w['max_price_mwh'].str.strip() == '']
        print(f"\n{name}: {len(w)} obs | missing hrs: {len(missing)} | "
              f"missing price: {len(empty_p)} | "
              f"scarcity hrs: {int((w['scarcity_flag_any_num']==1).sum())} | "
              f"avg resid demand: {w['residual_demand_mw_num'].mean():,.0f} MW")
        w['event'] = name
        w['event_time_hours'] = ((w['ts'] - c0) / pd.Timedelta(hours=1)).round().astype(int)
        frames.append(w[['event','event_time_hours'] + ORIG + NUMS])
    return pd.concat(frames, ignore_index=True)

# -----------------------------------------------------------
# 4. SUMMARY TABLE
# -----------------------------------------------------------

def build_summary(windows, centers):
    rows = []
    for name, c0 in centers.items():
        lo = c0 - pd.Timedelta(hours=72)
        hi = c0 + pd.Timedelta(hours=72)
        w  = windows[windows['event'] == name]
        a, b = EVENTS[name]
        rows.append({
            'event':                  name,
            'candidate_date_range':   f'{a} to {b}',
            't0_timestamp':           c0.strftime('%Y-%m-%d %H:%M'),
            'window_start':           lo.strftime('%Y-%m-%d %H:%M'),
            'window_end':             hi.strftime('%Y-%m-%d %H:%M'),
            'n_observations':         len(w),
            'n_scarcity_hours':       int((w['scarcity_flag_any_num']==1).sum()),
            'max_max_price_mwh':      w['max_price_mwh_num'].max(),
            'avg_residual_demand_mw': round(w['residual_demand_mw_num'].mean(), 2),
        })
    return pd.DataFrame(rows)

# -----------------------------------------------------------
# 5. FIGURE
# -----------------------------------------------------------

def plot_events(windows, centers, path):
    C_RD='#1f4e79'; C_PR='#c0392b'; C_SC='#f39c12'; C_MISS='#9aa0a6'
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    for ax, (name, c0) in zip(axes, centers.items()):
        w    = windows[windows['event']==name].sort_values('event_time_hours')
        x    = w['event_time_hours'].values
        rd   = w['residual_demand_mw_num'].values
        pr   = w['avg_price_mwh_num'].values
        scar = (w['scarcity_flag_any_num']==1).values
        miss = w['avg_price_mwh_num'].isna().values
        for xi in x[scar]: ax.axvspan(xi-.5, xi+.5, color=C_SC,   alpha=0.18, lw=0)
        for xi in x[miss]: ax.axvspan(xi-.5, xi+.5, color=C_MISS, alpha=0.45, lw=0)
        ax.axvline(0, color='black', ls='--', lw=1.4)
        ax.plot(x, rd, color=C_RD, lw=2.0)
        ax.set_ylim(20000, 70000)
        ax.set_ylabel('Residual demand (MW)', color=C_RD, fontsize=11)
        ax.tick_params(axis='y', labelcolor=C_RD)
        ax.set_title(f'{name}   —   t=0 at {c0.strftime("%Y-%m-%d %H:%M")}',
                     fontsize=12.5, fontweight='bold', pad=8)
        ax.grid(True, axis='x', alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(x, pr, color=C_PR, lw=1.8)
        ax2.set_ylim(-100, 1050)
        ax2.set_ylabel('Avg price ($/MWh)', color=C_PR, fontsize=11)
        ax2.tick_params(axis='y', labelcolor=C_PR)
        ax.set_xlim(-72, 72)
        ax.set_xticks(range(-72, 73, 12))
    axes[-1].set_xlabel('Event time (hours relative to t=0)', fontsize=12)
    fig.legend(handles=[
        Line2D([0],[0], color=C_RD,    lw=2,   label='Residual demand (MW)'),
        Line2D([0],[0], color=C_PR,    lw=1.8, label='Avg price ($/MWh)'),
        Line2D([0],[0], color='black', ls='--', lw=1.4, label='t = 0'),
        Patch(facecolor=C_SC,   alpha=0.18, label='Scarcity hour'),
        Patch(facecolor=C_MISS, alpha=0.45, label='Missing price (gap, not filled)'),
    ], loc='upper center', ncol=3, fontsize=10.5, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle('ERCOT grid stress: residual demand, price, and scarcity (±72h event study)',
                 fontsize=14, fontweight='bold', y=1.045)
    fig.text(0.5, -0.012,
        'Source: ercot_master_panel.csv. Values plotted as-is — no smoothing or interpolation. '
        'Grid conditions only; does not measure miner curtailment.',
        ha='center', fontsize=8.5, style='italic', color='#444')
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved: {path}")

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

if __name__ == '__main__':
    df      = load_panel(DATA_PATH)
    centers = select_centers(df)
    windows = extract_windows(df, centers)
    summary = build_summary(windows, centers)

    windows.to_csv(f'{OUTPUT_DIR}/event_window_data.csv',   index=False)
    summary.to_csv(f'{OUTPUT_DIR}/event_summary_table.csv', index=False)
    print(f"\nSummary:\n{summary.to_string(index=False)}")
    plot_events(windows, centers, f'{OUTPUT_DIR}/three_event_comparison.png')
    print("\nDone.")
