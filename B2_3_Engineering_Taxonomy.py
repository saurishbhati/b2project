"""
B2.3 — Engineering Flexibility Taxonomy of Computing and Industrial Loads
Author: Neev Naguboyina
Sub-project: B2.3 (Engineering Flexibility Taxonomy)
Group: B2 — Load Flexibility, AI Workloads, and Market Structure
Program: ASSIP 2026, George Mason University — Department of Finance

=============================================================
RESEARCH QUESTION
=============================================================
How fast, how deep, and at what restart cost can each major large
load actually move, and how do those physical parameters map to the
model's adjustment cost kappa? The taxonomy converts the paper's
"flexible vs. inflexible" split into a measured spectrum.

=============================================================
DATA
=============================================================
Source : Manufacturer specs, peer-reviewed demonstrations, and grid
         reliability filings (no ERCOT panel input required).

Load types:
  mining ASIC        - Antminer S19/S21-class Bitcoin miners
  AI training        - orchestrated GPU training clusters
  AI inference       - live model-serving workloads
  aluminum smelter   - electrolytic potline
  generic data center - hyperscale cloud, 99.999% uptime

Parameters per load type:
  ramp_seconds       - time to reduce power draw once instructed
  depth_pct          - max load reduction achievable, % of full load
  restart_penalty    - cost/delay to return to full output (tier)
  kappa_tier         - mapping to the model's adjustment cost kappa

=============================================================
METHODS
=============================================================
1. Assemble the parameter table for all five load types
2. Score claims L1, L2, A2 against the assembled parameters
3. Build the flexibility scorecard figure (ramp vs. depth, sized by
   restart cost, colored by kappa tier)
4. Emit the parameter table and claim-verification table as CSVs

=============================================================
CLAIMS TESTED
=============================================================
L1 - mining is memoryless, toggled machine-by-machine near-instantly
L2 - Antminer-class chips cycle in ~10-15 seconds
A2 - data centers cannot be easily shut down / restarted

=============================================================
KEY FINDING
=============================================================
The Bitcoin-vs-data-center wall is a workload-and-incentive wall, not
a warm-up wall. Mining flexes for free (kappa ~ 0), AI training flexes
cheaply when paid to, and inference plus uptime-bound cloud stay fixed
because of service contracts rather than physics.

=============================================================
OUTPUTS
=============================================================
  flexibility_scorecard.png     - ramp vs. depth scatter, sized by restart
  flexibility_parameters.csv    - one row per load type
  claim_verification.csv        - L1 / L2 / A2 confirm-refute table
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

KAPPA_COLORS = {
    'near-zero': '#1f4e79',
    'small':     '#2e86c1',
    'high':      '#c0392b',
    'nonlinear': '#f39c12',
    'contract':  '#7d3c98',
}

RESTART_SIZE = {
    'near-zero': 120,
    'low':       260,
    'moderate':  460,
    'severe':    820,
}

# -----------------------------------------------------------
# 1. PARAMETER TABLE
# -----------------------------------------------------------

def build_parameters():
    rows = [
        dict(load_type='mining ASIC',
             ramp_seconds=5, depth_pct=100, restart_penalty='near-zero',
             kappa_tier='near-zero', binding_constraint='pure economics'),
        dict(load_type='AI training',
             ramp_seconds=60, depth_pct=30, restart_penalty='low',
             kappa_tier='small', binding_constraint='chip opportunity cost'),
        dict(load_type='AI inference',
             ramp_seconds=120, depth_pct=10, restart_penalty='low',
             kappa_tier='high', binding_constraint='latency / SLA'),
        dict(load_type='aluminum smelter',
             ramp_seconds=30, depth_pct=25, restart_penalty='severe',
             kappa_tier='nonlinear', binding_constraint='thermal restart cost'),
        dict(load_type='generic data center',
             ramp_seconds=90, depth_pct=15, restart_penalty='moderate',
             kappa_tier='contract', binding_constraint='uptime SLA'),
    ]
    table = pd.DataFrame(rows)
    print("--- Flexibility parameter table ---")
    print(table.to_string(index=False))
    return table

# -----------------------------------------------------------
# 2. CLAIM VERIFICATION
# -----------------------------------------------------------

def verify_claims():
    rows = [
        dict(claim_id='L1',
             claim='mining is memoryless, toggled machine-by-machine near-instantly',
             verdict='holds',
             basis='S21 firmware sleep mode pauses hashing at near-zero draw; per-unit toggle'),
        dict(claim_id='L2',
             claim='Antminer-class chips cycle in ~10-15 seconds',
             verdict='holds with nuance',
             basis='power-state switch is seconds; full rated hash rate after cold reboot takes minutes'),
        dict(claim_id='A2',
             claim='data centers cannot be easily shut down / restarted',
             verdict='largely refuted for training; holds for inference / cloud',
             basis='25% cut sustained 3 hrs at full SLA (Phoenix); wall is workload type, not warm-up'),
    ]
    table = pd.DataFrame(rows)
    print("\n--- Claim verification (L1, L2, A2) ---")
    print(table.to_string(index=False))
    return table

# -----------------------------------------------------------
# 3. FLEXIBILITY SCORECARD FIGURE
# -----------------------------------------------------------

def plot_scorecard(params, path):
    fig, ax = plt.subplots(figsize=(11, 7))
    for _, r in params.iterrows():
        ax.scatter(r['ramp_seconds'], r['depth_pct'],
                   s=RESTART_SIZE[r['restart_penalty']],
                   color=KAPPA_COLORS[r['kappa_tier']],
                   alpha=0.75, edgecolor='black', linewidth=0.6, zorder=3)
        ax.annotate(r['load_type'],
                    (r['ramp_seconds'], r['depth_pct']),
                    xytext=(8, 6), textcoords='offset points',
                    fontsize=10.5, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlabel('Ramp-down time (seconds, log scale)', fontsize=11)
    ax.set_ylabel('Curtailment depth (% of full load)', fontsize=11)
    ax.set_ylim(-5, 110)
    ax.set_title('Flexibility scorecard: ramp vs. depth, sized by restart cost, colored by adjustment cost kappa',
                 fontsize=12.5, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.25, zorder=0)

    kappa_handles = [plt.Line2D([0], [0], marker='o', color='w',
                     markerfacecolor=c, markersize=11, label=f'kappa: {k}')
                     for k, c in KAPPA_COLORS.items()]
    size_handles = [plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor='gray', markersize=np.sqrt(s) / 1.6,
                    label=f'restart: {lbl}')
                    for lbl, s in RESTART_SIZE.items()]
    ax.legend(handles=kappa_handles + size_handles, fontsize=9,
              loc='center right', framealpha=0.95)

    fig.text(0.5, -0.02,
        'Source: manufacturer specs, peer-reviewed demonstrations (Phoenix, Nature Energy 2025), '
        'and NERC / EPRI reliability filings. Parameters are order-of-magnitude engineering estimates.',
        ha='center', fontsize=8.5, style='italic', color='#444')
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved: {path}")

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

if __name__ == '__main__':
    params = build_parameters()
    claims = verify_claims()

    params.to_csv(f'{OUTPUT_DIR}/flexibility_parameters.csv', index=False)
    claims.to_csv(f'{OUTPUT_DIR}/claim_verification.csv', index=False)
    plot_scorecard(params, f'{OUTPUT_DIR}/flexibility_scorecard.png')
    print("\nDone.")
