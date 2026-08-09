#!/usr/bin/env python3
"""Visualize power load time series for any raw or processed dataset.

Usage:
  python visualize_power.py --input data/processed/eld_ind.csv
  python visualize_power.py --input data/processed/eld_ind.csv --cols MT_001,MT_050,MT_100
  python visualize_power.py --input data/processed/lcl_res.csv --cols MAC000001,MAC000002
  python visualize_power.py --input data/processed/steel_ind.csv --full
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "processed" / "figures"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
})

SKIP_COLS = {
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "month_sin", "month_cos", "category_id", "tariff", "Load_Type",
    "load_type", "LCLid", "Date", "Time", "DateTime", "NSM", "Day",
}

DT_CANDIDATES = ["datetime", "DateTime", "date", "timestamp", "time"]

# Fixed column selections for known datasets (deterministic across runs)
FIXED_SELECTION = {
    "eld_ind": [
        "MT_001", "MT_050", "MT_100", "MT_150", "MT_200", "MT_250",
    ],
    "steel_ind": [
        "Usage_kWh", "lagging_reactive_power", "leading_reactive_power",
        "co2", "lagging_pf", "leading_pf",
    ],
    "tetouan_city": [
        "load_zone1", "load_zone2", "load_zone3",
        "temperature", "humidity", "wind_speed",
    ],
    "household_res": [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2",
        "Sub_metering_3",
    ],
    "lcl_res": [
        "MAC000001", "MAC000002", "MAC000003", "MAC000004", "MAC000005",
        "MAC000006",
    ],
}


def _detect_datetime_col(df: pd.DataFrame) -> str:
    for c in DT_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            try:
                pd.to_datetime(df[c], errors="raise")
                return c
            except (ValueError, Exception):
                pass
    raise ValueError(f"Cannot detect datetime column. Candidates: {DT_CANDIDATES}")


def _detect_power_cols(df: pd.DataFrame, dt_col: str) -> list:
    numeric = df.select_dtypes(include=[np.number]).columns
    cols = [c for c in numeric if c not in SKIP_COLS and c != dt_col]
    if not cols:
        raise ValueError("No plottable numeric columns found")
    return cols


def _resolve_columns(df: pd.DataFrame, dt_col: str, power_cols: list,
                     dataset_name: str, user_cols: list | None,
                     n_seq: int) -> tuple:
    """Resolve which columns/users to plot.

    Priority: --cols arg > FIXED_SELECTION[dataset] > auto (evenly spaced / top-N).
    Returns (selected, is_long_format, group_col).
    """
    is_long = "LCLid" in df.columns

    if user_cols:
        # User explicitly specified columns/IDs
        if is_long:
            valid = [u for u in user_cols if u in df["LCLid"].unique()]
        else:
            valid = [c for c in user_cols if c in df.columns]
        if not valid:
            print(f"WARNING: none of {user_cols} found, falling back to auto")
        else:
            return valid[:n_seq], is_long, "LCLid" if is_long else None

    # Use fixed selection from FIXED_SELECTION dict
    fixed = FIXED_SELECTION.get(dataset_name)
    if fixed:
        if is_long:
            valid = [u for u in fixed if u in df["LCLid"].unique()]
        else:
            valid = [c for c in fixed if c in df.columns]
        if valid:
            return valid[:n_seq], is_long, "LCLid" if is_long else None

    # Auto: top-N for long, evenly spaced for wide
    if is_long:
        user_sizes = df.groupby("LCLid").size()
        top = user_sizes.nlargest(min(n_seq, len(user_sizes))).index.tolist()
        return top, True, "LCLid"

    if len(power_cols) <= n_seq:
        return power_cols, False, None
    idxs = np.linspace(0, len(power_cols) - 1, n_seq, dtype=int)
    return [power_cols[i] for i in idxs], False, None


def plot_dataset(csv_path: str, n_seq: int = 6, n_weeks: int = 2,
                 full_range: bool = False, output: str | None = None,
                 user_cols: list | None = None):
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {csv_path} ({len(df)} rows x {len(df.columns)} cols)")

    dt_col = _detect_datetime_col(df)
    t = pd.to_datetime(df[dt_col])
    power_cols = _detect_power_cols(df, dt_col)
    dataset_name = Path(csv_path).stem

    selected, is_long, group_col = _resolve_columns(
        df, dt_col, power_cols, dataset_name, user_cols, n_seq)

    if not selected:
        print("ERROR: no valid columns/IDs to plot")
        return

    print(f"  plotting: {selected}")

    if full_range:
        mask = slice(None)
        title_suffix = "full range"
    else:
        t_min = t.min()
        t_max = t_min + pd.Timedelta(weeks=n_weeks)
        mask = (t >= t_min) & (t < t_max)
        title_suffix = f"first {n_weeks} week(s)"

    if is_long:
        _plot_long(df, dt_col, mask, selected, group_col, dataset_name,
                   title_suffix, output)
    else:
        _plot_wide(df, dt_col, mask, selected, dataset_name, title_suffix, output)


def _plot_wide(df, dt_col, mask, cols, dataset_name, title_suffix, output):
    t = pd.to_datetime(df[dt_col])
    fig, ax = plt.subplots(figsize=(14, 5))
    for c in cols:
        s = pd.to_numeric(df[c].loc[mask], errors="coerce")
        ax.plot(t.loc[mask], s.values, linewidth=0.5, label=c, alpha=0.85)

    ax.set_title(f"{dataset_name} — {len(cols)} sequences ({title_suffix})")
    ax.set_ylabel("Value")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    if len(cols) <= 10:
        ax.legend(fontsize=7, ncol=min(4, len(cols)))
    fig.tight_layout()

    out_path = output or str(OUT_DIR / f"{dataset_name}_power.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → {out_path}")


def _plot_long(df, dt_col, mask, user_ids, group_col, dataset_name,
               title_suffix, output):
    val_cols = _detect_power_cols(df, dt_col)
    val_col = val_cols[0] if val_cols else "KWH"

    fig, ax = plt.subplots(figsize=(14, 5))
    for uid in user_ids:
        idx = (df[group_col] == uid) & mask
        sub = df.loc[idx].sort_values(dt_col)
        if sub.empty:
            continue
        s = pd.to_numeric(sub[val_col], errors="coerce")
        ax.plot(pd.to_datetime(sub[dt_col]), s.values, linewidth=0.5,
                label=str(uid), alpha=0.85)

    ax.set_title(f"{dataset_name} — {len(user_ids)} {group_col}s ({title_suffix})")
    ax.set_ylabel(val_col)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    if len(user_ids) <= 10:
        ax.legend(fontsize=7, ncol=min(4, len(user_ids)))
    fig.tight_layout()

    out_path = output or str(OUT_DIR / f"{dataset_name}_power.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot power time series from any CSV dataset")
    ap.add_argument("--input", required=True,
                    help="Path to CSV file (raw or processed)")
    ap.add_argument("--cols", default=None,
                    help="Comma-separated column names or user IDs to plot "
                         "(overrides auto-selection)")
    ap.add_argument("--n", type=int, default=6,
                    help="Number of sequences to plot (max 10, default 6)")
    ap.add_argument("--weeks", type=int, default=2,
                    help="Number of weeks to show (default 2)")
    ap.add_argument("--full", action="store_true",
                    help="Plot full time range")
    ap.add_argument("--output", default=None,
                    help="Output figure path (auto-generated if omitted)")
    args = ap.parse_args()

    n_seq = min(args.n, 10)
    user_cols = [c.strip() for c in args.cols.split(",")] if args.cols else None

    try:
        plot_dataset(args.input, n_seq=n_seq, n_weeks=args.weeks,
                     full_range=args.full, output=args.output,
                     user_cols=user_cols)
    except FileNotFoundError:
        print(f"ERROR: file not found: {args.input}")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
