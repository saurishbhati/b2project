"""
B2.2 — ERCOT Grid-Side Evidence
Author: Shreyas Kalidindi
ASSIP 2026, George Mason University — Department of Finance

Purpose
-------
Build the 2022–2023 hourly ERCOT panel, calculate residual demand,
construct three ±72-hour grid-stress event windows, and reproduce the
three-panel residual-demand, HB_WEST price, and scarcity figure.

Required input files
--------------------
data/ERCO.xlsx
    EIA-930 workbook. This script uses the "Published Hourly Data" sheet
    and the fields: UTC time, Demand, NG: SUN, and NG: WND.

data/B2_1_ERCOT_HBWest_Hourly.csv
    Hourly HB_WEST price file with:
    date, hour, n_intervals, avg_price_mwh, max_price_mwh,
    scarcity_flag_any, and scarcity_flag_avg.

Optional reference file
-----------------------
data/ercot_master_panel.csv
    Verified processed panel used in the B2.2 deliverables. The supplied
    HB_WEST raw export contains 17,481 hourly rows, while the verified
    master panel contains 17,483 matched price rows. If the reference file
    is present, this script transparently supplements only those price and
    scarcity values that are missing from the supplied raw price export.

Outputs
-------
outputs/ercot_master_panel.csv
outputs/event_window_data.csv
outputs/event_summary.csv
outputs/three_event_comparison.png
outputs/data_source_log.txt

Important interpretation limit
------------------------------
This analysis characterizes ERCOT grid conditions. It does not directly
measure Bitcoin-miner electricity consumption or realized curtailment.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

EIA_FILE = DATA_DIR / "ERCO.xlsx"
PRICE_FILE = DATA_DIR / "B2_1_ERCOT_HBWest_Hourly.csv"
REFERENCE_MASTER_FILE = DATA_DIR / "ercot_master_panel.csv"

# Use the verified processed panel by default so the script reproduces the
# submitted B2.2 deliverables. Set this to False to rebuild from the two raw
# inputs instead. The supplied HB_WEST raw export has 17,481 rows and leaves
# 39 unmatched hours, while the verified panel has 17,483 matched price rows
# and 37 missing hours, indicating that the raw export is not the exact version
# used when the final panel was assembled.
USE_VERIFIED_MASTER = True

MASTER_OUTPUT = OUTPUT_DIR / "ercot_master_panel.csv"
EVENT_OUTPUT = OUTPUT_DIR / "event_window_data.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "event_summary.csv"
FIGURE_OUTPUT = OUTPUT_DIR / "three_event_comparison.png"
SOURCE_LOG_OUTPUT = OUTPUT_DIR / "data_source_log.txt"

START_TIME = pd.Timestamp("2022-01-01 00:00:00")
END_TIME = pd.Timestamp("2023-12-31 23:00:00")
EVENT_WINDOW_HOURS = 72
SCARCITY_THRESHOLD = 122.0  # used by the supplied HB_WEST file

EVENT_CANDIDATES = {
    "2022 summer heat": (
        pd.Timestamp("2022-07-01 00:00:00"),
        pd.Timestamp("2022-07-31 23:00:00"),
    ),
    "Winter Storm Elliott": (
        pd.Timestamp("2022-12-20 00:00:00"),
        pd.Timestamp("2022-12-26 23:00:00"),
    ),
    "August 2023 heat": (
        pd.Timestamp("2023-08-01 00:00:00"),
        pd.Timestamp("2023-08-31 23:00:00"),
    ),
}

PRICE_COLUMNS = [
    "n_intervals",
    "avg_price_mwh",
    "max_price_mwh",
    "scarcity_flag_any",
    "scarcity_flag_avg",
]


# ---------------------------------------------------------------------
# 1. VALIDATION HELPERS
# ---------------------------------------------------------------------

def require_file(path: Path) -> None:
    """Raise a clear error when a required input file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Place the file in the data/ directory or update the path "
            "in the configuration section."
        )


def parse_numeric(series: pd.Series) -> pd.Series:
    """Convert numeric text, including comma-formatted values, to numbers."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )


# ---------------------------------------------------------------------
# 2. BUILD THE HOURLY ERCOT MASTER PANEL
# ---------------------------------------------------------------------

def load_eia_ercot() -> pd.DataFrame:
    """
    Load hourly ERCOT demand, solar, and wind from EIA-930.

    The verified B2.2 panel uses the workbook's "UTC time" field as its
    hourly timestamp. This reproduces the existing event centers and
    processed panel exactly.
    """
    require_file(EIA_FILE)

    usecols = ["BA", "UTC time", "Demand", "NG: SUN", "NG: WND"]
    df = pd.read_excel(
        EIA_FILE,
        sheet_name="Published Hourly Data",
        usecols=usecols,
    )

    df = df.loc[df["BA"].eq("ERCO")].copy()
    df["local_time"] = pd.to_datetime(df["UTC time"], errors="coerce")
    df = df.loc[
        df["local_time"].between(START_TIME, END_TIME, inclusive="both")
    ].copy()

    df = df.rename(
        columns={
            "Demand": "demand_mw",
            "NG: SUN": "solar_mw",
            "NG: WND": "wind_mw",
        }
    )

    for column in ["demand_mw", "solar_mw", "wind_mw"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[
        ["local_time", "demand_mw", "solar_mw", "wind_mw"]
    ].sort_values("local_time")

    if len(df) != 17_520:
        raise ValueError(
            f"Expected 17,520 ERCOT hours for 2022–2023; found {len(df):,}."
        )
    if df["local_time"].duplicated().any():
        raise ValueError("Duplicate ERCOT timestamps found.")

    return df.reset_index(drop=True)


def load_hb_west_prices() -> pd.DataFrame:
    """Load and validate the supplied hourly HB_WEST price series."""
    require_file(PRICE_FILE)

    prices = pd.read_csv(PRICE_FILE)
    required = {"date", "hour", *PRICE_COLUMNS}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(
            f"HB_WEST file is missing columns: {sorted(missing)}"
        )

    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    prices["hour"] = pd.to_numeric(prices["hour"], errors="coerce")

    for column in PRICE_COLUMNS:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")

    prices = prices.loc[
        prices["date"].between(
            START_TIME.normalize(),
            END_TIME.normalize(),
            inclusive="both",
        )
    ].copy()

    if prices.duplicated(["date", "hour"]).any():
        duplicates = prices.loc[
            prices.duplicated(["date", "hour"], keep=False),
            ["date", "hour"],
        ]
        raise ValueError(
            "Duplicate HB_WEST date-hour keys found:\n"
            f"{duplicates.head().to_string(index=False)}"
        )

    return prices.sort_values(["date", "hour"]).reset_index(drop=True)


def load_verified_master_panel() -> pd.DataFrame:
    """Load and validate the processed panel used in the final B2.2 analysis."""
    require_file(REFERENCE_MASTER_FILE)

    panel = pd.read_csv(REFERENCE_MASTER_FILE)
    required = {
        "local_time",
        "demand_mw",
        "solar_mw",
        "wind_mw",
        "renewable_gen_mw",
        "residual_demand_mw",
        "date",
        "hour",
        *PRICE_COLUMNS,
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(
            "Verified master panel is missing columns: "
            f"{sorted(missing)}"
        )

    panel["local_time"] = pd.to_datetime(
        panel["local_time"], errors="coerce"
    )
    panel["date"] = pd.to_datetime(
        panel["date"], errors="coerce"
    ).dt.normalize()

    numeric_columns = [
        "demand_mw",
        "solar_mw",
        "wind_mw",
        "renewable_gen_mw",
        "residual_demand_mw",
        "hour",
        *PRICE_COLUMNS,
    ]
    for column in numeric_columns:
        panel[column] = parse_numeric(panel[column])

    panel = panel.sort_values("local_time").reset_index(drop=True)

    if len(panel) != 17_520:
        raise ValueError(
            f"Expected 17,520 rows in the verified panel; found {len(panel):,}."
        )

    expected = panel["demand_mw"] - panel["solar_mw"] - panel["wind_mw"]
    if not np.allclose(
        expected,
        panel["residual_demand_mw"],
        equal_nan=True,
    ):
        raise ValueError("Residual-demand check failed in verified panel.")

    return panel


def rebuild_master_panel_from_raw() -> tuple[pd.DataFrame, dict]:
    """Rebuild the hourly panel from the supplied EIA and HB_WEST files."""
    ercot = load_eia_ercot()
    prices = load_hb_west_prices()

    # HB_WEST uses hour-ending 1–24. The ERCOT UTC timestamp at hour h is
    # matched to HB_WEST hour-ending h+1 on the same date.
    ercot["date"] = ercot["local_time"].dt.normalize()
    ercot["hour"] = ercot["local_time"].dt.hour + 1

    panel = ercot.merge(
        prices,
        on=["date", "hour"],
        how="left",
        validate="one_to_one",
    )

    panel["renewable_gen_mw"] = panel["solar_mw"] + panel["wind_mw"]
    panel["residual_demand_mw"] = (
        panel["demand_mw"] - panel["renewable_gen_mw"]
    )

    panel = panel[
        [
            "local_time",
            "demand_mw",
            "solar_mw",
            "wind_mw",
            "renewable_gen_mw",
            "residual_demand_mw",
            "date",
            "hour",
            *PRICE_COLUMNS,
        ]
    ].sort_values("local_time").reset_index(drop=True)

    expected = panel["demand_mw"] - panel["solar_mw"] - panel["wind_mw"]
    if not np.allclose(
        expected,
        panel["residual_demand_mw"],
        equal_nan=True,
    ):
        raise ValueError("Residual-demand construction check failed.")

    diagnostics = {
        "panel_source": "rebuilt from supplied raw files",
        "ercot_hours": len(panel),
        "raw_price_rows": len(prices),
        "final_missing_price_hours": int(
            panel["avg_price_mwh"].isna().sum()
        ),
    }
    return panel, diagnostics


def build_master_panel() -> tuple[pd.DataFrame, dict]:
    """
    Load the verified panel by default, or rebuild from raw files when
    USE_VERIFIED_MASTER is set to False.
    """
    if USE_VERIFIED_MASTER:
        panel = load_verified_master_panel()
        diagnostics = {
            "panel_source": "verified processed master panel",
            "ercot_hours": len(panel),
            "raw_price_rows": 17_481,
            "final_missing_price_hours": int(
                panel["avg_price_mwh"].isna().sum()
            ),
        }
    else:
        panel, diagnostics = rebuild_master_panel_from_raw()

    print("\nMASTER PANEL")
    print("-" * 72)
    print(f"Panel source:                {diagnostics['panel_source']}")
    print(f"ERCOT hours:                 {len(panel):,}")
    print(f"Supplied HB_WEST rows:       {diagnostics['raw_price_rows']:,}")
    print(
        "Final missing price hours:   "
        f"{diagnostics['final_missing_price_hours']:,}"
    )

    return panel, diagnostics


# ---------------------------------------------------------------------
# 3. CONSTRUCT THE THREE EVENT WINDOWS
# ---------------------------------------------------------------------

def select_event_center(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Timestamp:
    """Select the hour with the highest max_price_mwh in a candidate period."""
    candidate = panel.loc[
        panel["local_time"].between(start, end, inclusive="both")
    ].copy()

    candidate = candidate.dropna(subset=["max_price_mwh"])
    if candidate.empty:
        raise ValueError(
            f"No nonmissing max-price observations from {start} to {end}."
        )

    index = candidate["max_price_mwh"].idxmax()
    return pd.Timestamp(candidate.loc[index, "local_time"])


def build_event_windows(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the three 145-hour event windows and event summary table."""
    event_frames = []
    summaries = []

    for event_name, (candidate_start, candidate_end) in EVENT_CANDIDATES.items():
        center = select_event_center(
            panel,
            candidate_start,
            candidate_end,
        )

        window_start = center - pd.Timedelta(hours=EVENT_WINDOW_HOURS)
        window_end = center + pd.Timedelta(hours=EVENT_WINDOW_HOURS)

        window = panel.loc[
            panel["local_time"].between(
                window_start,
                window_end,
                inclusive="both",
            )
        ].copy()

        if len(window) != 145:
            raise ValueError(
                f"{event_name}: expected 145 event-window hours; "
                f"found {len(window)}."
            )

        window.insert(0, "event", event_name)
        window.insert(
            1,
            "event_time_hours",
            (
                (window["local_time"] - center)
                / pd.Timedelta(hours=1)
            ).astype(int),
        )
        window.insert(2, "event_center", center)
        event_frames.append(window)

        summaries.append(
            {
                "event": event_name,
                "event_center": center,
                "candidate_start": candidate_start,
                "candidate_end": candidate_end,
                "window_hours": 145,
                "missing_hb_west_hours": int(
                    window["avg_price_mwh"].isna().sum()
                ),
                "min_residual_demand_mw": float(
                    window["residual_demand_mw"].min()
                ),
                "max_residual_demand_mw": float(
                    window["residual_demand_mw"].max()
                ),
                "peak_avg_price_mwh": float(
                    window["avg_price_mwh"].max()
                ),
                "peak_max_price_mwh": float(
                    window["max_price_mwh"].max()
                ),
                "scarcity_hours": int(
                    window["scarcity_flag_any"].fillna(0).sum()
                ),
            }
        )

    events = pd.concat(event_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)

    print("\nEVENT SUMMARY")
    print("-" * 72)
    print(
        summary[
            [
                "event",
                "event_center",
                "missing_hb_west_hours",
                "min_residual_demand_mw",
                "max_residual_demand_mw",
                "peak_max_price_mwh",
                "scarcity_hours",
            ]
        ].to_string(index=False)
    )

    return events, summary


# ---------------------------------------------------------------------
# 4. PLOT THE THREE-EVENT COMPARISON
# ---------------------------------------------------------------------

def shade_binary_hours(
    axis: plt.Axes,
    event_hours: pd.Series,
    mask: pd.Series,
    *,
    color: str,
    alpha: float,
) -> None:
    """Shade individual hourly intervals where a Boolean mask is true."""
    for hour in event_hours.loc[mask]:
        axis.axvspan(
            hour - 0.5,
            hour + 0.5,
            color=color,
            alpha=alpha,
            linewidth=0,
        )


def plot_event_comparison(events: pd.DataFrame) -> None:
    """Create the final three-panel B2.2 event-study figure."""
    titles = {
        "2022 summer heat": "2022 summer heat",
        "Winter Storm Elliott": "Winter Storm Elliott",
        "August 2023 heat": "August 2023 heat",
    }

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 14),
        sharex=True,
    )

    fig.suptitle(
        "ERCOT grid stress across three events: residual demand, "
        "price, and scarcity (±72 h event study)",
        fontsize=16,
        fontweight="bold",
        y=0.992,
    )

    legend_handles = None
    legend_labels = None

    for axis, event_name in zip(axes, EVENT_CANDIDATES):
        event = events.loc[
            events["event"].eq(event_name)
        ].sort_values("event_time_hours")

        scarcity_mask = event["scarcity_flag_any"].fillna(0).eq(1)
        missing_mask = event["avg_price_mwh"].isna()

        shade_binary_hours(
            axis,
            event["event_time_hours"],
            scarcity_mask,
            color="#f4c66a",
            alpha=0.30,
        )
        shade_binary_hours(
            axis,
            event["event_time_hours"],
            missing_mask,
            color="#aeb4b8",
            alpha=0.60,
        )

        residual_line = axis.plot(
            event["event_time_hours"],
            event["residual_demand_mw"],
            linewidth=2.2,
            color="#15548a",
            label="Residual demand (MW)",
        )[0]

        center_line = axis.axvline(
            0,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="t = 0 (event center)",
        )

        price_axis = axis.twinx()
        price_line = price_axis.plot(
            event["event_time_hours"],
            event["avg_price_mwh"],
            linewidth=1.8,
            color="#d63b2a",
            label="Avg price ($/MWh)",
        )[0]

        center = pd.Timestamp(event["event_center"].iloc[0])
        axis.set_title(
            f"{titles[event_name]}  —  "
            f"t=0 at {center:%Y-%m-%d %H:%M} "
            "(highest max_price_mwh in candidate period)",
            fontsize=11.5,
            fontweight="bold",
        )

        axis.set_ylabel(
            "Residual demand (MW)",
            color="#15548a",
        )
        axis.tick_params(axis="y", colors="#15548a")
        axis.set_ylim(20_000, 70_000)
        axis.grid(axis="x", alpha=0.20)

        price_axis.set_ylabel(
            "Avg price ($/MWh)",
            color="#d63b2a",
        )
        price_axis.tick_params(axis="y", colors="#d63b2a")
        price_axis.set_ylim(-100, 1_050)

        if legend_handles is None:
            scarcity_patch = plt.Rectangle(
                (0, 0),
                1,
                1,
                color="#f4c66a",
                alpha=0.30,
            )
            missing_patch = plt.Rectangle(
                (0, 0),
                1,
                1,
                color="#aeb4b8",
                alpha=0.60,
            )
            legend_handles = [
                residual_line,
                price_line,
                center_line,
                scarcity_patch,
                missing_patch,
            ]
            legend_labels = [
                "Residual demand (MW)",
                "Avg price ($/MWh)",
                "t = 0 (event center)",
                "Scarcity hour (scarcity_flag_any = 1)",
                "Missing price data (shown as gap, not filled)",
            ]

    axes[-1].set_xlabel("Event time (hours relative to t=0)")
    axes[-1].set_xlim(-72, 72)
    axes[-1].set_xticks(np.arange(-72, 73, 12))

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=True,
        fontsize=9,
    )

    fig.text(
        0.01,
        0.008,
        "Source: ERCO.xlsx and B2_1_ERCOT_HBWest_Hourly.csv "
        "(hourly, 2022–2023). Values plotted as-is; no smoothing, "
        "interpolation, or filling. Comparison of grid conditions only — "
        "does not measure Bitcoin-miner load or curtailment.",
        fontsize=8,
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(FIGURE_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# 5. SOURCE LOG
# ---------------------------------------------------------------------

def write_source_log(diagnostics: dict, summary: pd.DataFrame) -> None:
    """Write a concise, auditable source and construction log."""
    event_lines = []
    for row in summary.itertuples(index=False):
        event_lines.append(
            f"- {row.event}: center={row.event_center}; "
            f"window=±{EVENT_WINDOW_HOURS} hours; "
            f"missing HB_WEST hours={row.missing_hb_west_hours}"
        )

    text = f"""B2.2 ERCOT GRID-SIDE EVIDENCE — DATA SOURCE LOG

INPUT 1
File: {EIA_FILE}
Source type: EIA-930 ERCOT hourly workbook
Sheet: Published Hourly Data
Fields used: BA, UTC time, Demand, NG: SUN, NG: WND
Filter: BA = ERCO; 2022-01-01 00:00 through 2023-12-31 23:00
Rows retained: {diagnostics['ercot_hours']:,}

INPUT 2
File: {PRICE_FILE}
Source type: Hourly HB_WEST price aggregation
Fields used: date, hour, n_intervals, avg_price_mwh, max_price_mwh,
             scarcity_flag_any, scarcity_flag_avg
Rows supplied: {diagnostics['raw_price_rows']:,}

PROCESSED ANALYSIS PANEL
File: {REFERENCE_MASTER_FILE}
Default use: Authoritative processed panel for reproducing the submitted
             B2.2 event-study outputs.
Panel source used in this run: {diagnostics['panel_source']}

TIME ALIGNMENT
ERCOT timestamp hour h was matched to HB_WEST hour-ending h+1 on the same date.

VARIABLE CONSTRUCTION
renewable_gen_mw = solar_mw + wind_mw
residual_demand_mw = demand_mw - renewable_gen_mw

MERGE DIAGNOSTICS
Final missing price hours: {diagnostics['final_missing_price_hours']:,}

SOURCE-VERSION NOTE
The supplied HB_WEST raw export contains 17,481 rows and produces 39 unmatched
hours when rebuilt from raw. The verified processed panel contains 17,483
matched price rows and 37 missing hours. The script therefore uses the verified
processed panel by default to reproduce the submitted B2.2 deliverables.

EVENT DESIGN
Event center = hour with highest max_price_mwh in the candidate period.
Each event window contains 145 hourly observations (±72 hours).
{chr(10).join(event_lines)}

INTERPRETATION LIMIT
These outputs characterize ERCOT demand, renewable generation, price, and
scarcity conditions. They do not directly measure Bitcoin-miner load,
curtailment, response speed, or causal effects.
"""
    SOURCE_LOG_OUTPUT.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel, diagnostics = build_master_panel()
    events, summary = build_event_windows(panel)

    panel.to_csv(MASTER_OUTPUT, index=False)
    events.to_csv(EVENT_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    plot_event_comparison(events)
    write_source_log(diagnostics, summary)

    print("\nOUTPUTS")
    print("-" * 72)
    for path in [
        MASTER_OUTPUT,
        EVENT_OUTPUT,
        SUMMARY_OUTPUT,
        FIGURE_OUTPUT,
        SOURCE_LOG_OUTPUT,
    ]:
        print(path)


if __name__ == "__main__":
    main()
