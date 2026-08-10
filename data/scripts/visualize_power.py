#!/usr/bin/env python3
"""Visualize power load time series for any raw or processed dataset.

Labels and features are plotted in separate figures by group.
Group definitions are hardcoded per dataset (§DATASET_PLOT_COLS).

Usage:
  python visualize_power.py --input data/processed/eld_ind.csv
  python visualize_power.py --input data/raw/steel_ind/Steel_industry_data.csv
  python visualize_power.py --input data/processed/steel_ind.csv --full
  python visualize_power.py --input data/processed/lcl_res.csv --n 4 --weeks 1
  python visualize_power.py --input data/raw/tetouan_city/Tetuan_City_power_consumption.csv --output my_plot
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "figures"
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

# ── Hardcoded column groups per dataset ──────────────────────────────
# Each dataset has named groups; each group has "proc" and "raw" column lists.
# Labels and features are separated; features on different scales are split.
DATASET_PLOT_COLS = {
    "eld_ind": {
        "labels": {
            "proc": ["MT_200", "MT_210", "MT_220", "MT_230", "MT_240", "MT_250"],
            "raw":  ["MT_200", "MT_210", "MT_220", "MT_230", "MT_240", "MT_250"],
        },
    },
    "lcl_res": {
        "labels": {
            "proc": ["MAC000145", "MAC000149", "MAC000150",
                     "MAC000151", "MAC000152", "MAC000153"],
            "raw":  ["MAC000145", "MAC000149", "MAC000150",
                     "MAC000151", "MAC000152", "MAC000153"],
        },
    },
    "steel_ind": {
        "labels": {
            "proc": ["Usage_kWh"],
            "raw":  ["Usage_kWh"],
        },
        "features_reactive": {
            "proc": ["lagging_reactive_power", "leading_reactive_power"],
            "raw":  ["Lagging_Current_Reactive.Power_kVarh",
                     "Leading_Current_Reactive_Power_kVarh"],
        },
        "features_co2_pf": {
            "proc": ["co2", "lagging_pf", "leading_pf"],
            "raw":  ["CO2(tCO2)", "Lagging_Current_Power_Factor",
                     "Leading_Current_Power_Factor"],
        },
    },
    "tetouan_city": {
        "labels": {
            "proc": ["load_zone1", "load_zone2", "load_zone3"],
            "raw":  ["Zone 1 Power Consumption", "Zone 2  Power Consumption",
                     "Zone 3  Power Consumption"],
        },
        "features_weather": {
            "proc": ["temperature", "humidity", "wind_speed",
                     "general_diffuse_flow", "diffuse_flow"],
            "raw":  ["Temperature", "Humidity", "Wind Speed",
                     "general diffuse flows", "diffuse flows"],
        },
    },
}

# Unit conversions for raw data (applied before plotting).
RAW_CONVERSIONS = {
    ("steel_ind", "Lagging_Current_Power_Factor"): lambda s: pd.to_numeric(s, errors="coerce") / 100.0,
    ("steel_ind", "Leading_Current_Power_Factor"): lambda s: pd.to_numeric(s, errors="coerce") / 100.0,
}


def _parse_datetime(series: pd.Series):
    """Parse datetime, trying dayfirst=False then dayfirst=True; keep fewer NaT."""
    s_default = pd.to_datetime(series, errors="coerce")
    if s_default.isna().mean() < 0.1:
        return s_default
    s_dayfirst = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return s_default if s_default.notna().sum() >= s_dayfirst.notna().sum() else s_dayfirst


def _detect_dataset_id(csv_path: str) -> str:
    path = Path(csv_path).resolve()
    if "raw" in path.parts:
        return path.parent.name
    return path.stem


def _detect_datetime_col(df: pd.DataFrame) -> str:
    for c in DT_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            try:
                s = _parse_datetime(df[c])
                if s.notna().sum() > len(df) * 0.8:
                    return c
            except Exception:
                pass
    raise ValueError(f"Cannot detect datetime column. Candidates: {DT_CANDIDATES}")


def _resolve_groups(df: pd.DataFrame, dataset_id: str, is_raw: bool,
                    user_cols: list | None, n_seq: int) -> list[tuple[str, list]]:
    """Return [(group_name, [col_names]), ...] to plot.

    If --cols is given, all resolved columns go into a single "custom" group.
    Otherwise, use the hardcoded groups from DATASET_PLOT_COLS.
    """
    if user_cols:
        # Resolve each user-specified name against all groups' proc+raw lists
        groups = DATASET_PLOT_COLS.get(dataset_id, {})
        resolved = []
        for c in user_cols:
            if c in df.columns:
                resolved.append(c)
                continue
            # Try raw→proc or proc→raw translation across all groups
            found = False
            for grp in groups.values():
                proc_list = grp.get("proc", [])
                raw_list = grp.get("raw", [])
                if c in raw_list and len(raw_list) == len(proc_list):
                    idx = raw_list.index(c)
                    if proc_list[idx] in df.columns:
                        resolved.append(proc_list[idx])
                        found = True
                        break
                elif c in proc_list and len(raw_list) == len(proc_list):
                    idx = proc_list.index(c)
                    if raw_list[idx] in df.columns:
                        resolved.append(raw_list[idx])
                        found = True
                        break
            if not found:
                print(f"  WARNING: column '{c}' not found in {dataset_id}")
        if resolved:
            return [("custom", resolved[:n_seq])]
        return []

    # No --cols: use hardcoded groups
    groups = DATASET_PLOT_COLS.get(dataset_id, {})
    if not groups:
        print(f"  WARNING: no hardcoded columns for '{dataset_id}'")
        return []

    result = []
    for grp_name, grp_cols in groups.items():
        # Prefer key matching data source
        primary_key = "raw" if is_raw else "proc"
        fallback_key = "proc" if is_raw else "raw"
        cols = None
        for key in (primary_key, fallback_key):
            candidates = grp_cols.get(key, [])
            valid = [c for c in candidates if c in df.columns]
            if valid:
                cols = valid[:n_seq]
                break
        if cols:
            result.append((grp_name, cols))
        else:
            print(f"  WARNING: no columns found for group '{grp_name}' in {dataset_id}")
    return result


def _make_output_path(output: str | None, dataset_id: str, group_name: str,
                      is_raw: bool) -> str:
    """Build output path: {base}_{group}.png or {OUT_DIR}/{dataset}_{src}_{group}.png."""
    src_tag = "raw" if is_raw else "proc"
    if output:
        p = Path(output)
        out_dir = p.parent
        out_dir.mkdir(exist_ok=True)
        return str(out_dir / f"{p.stem}_{group_name}.png")
    return str(OUT_DIR / f"{dataset_id}_{src_tag}_{group_name}.png")


def plot_dataset(csv_path: str, n_seq: int = 6, n_weeks: int = 2,
                 full_range: bool = False, output: str | None = None,
                 user_cols: list | None = None):
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {csv_path} ({len(df)} rows x {len(df.columns)} cols)")

    dataset_id = _detect_dataset_id(csv_path)
    is_raw = "raw" in Path(csv_path).resolve().parts

    # Apply raw unit conversions (e.g. power factor % → 0-1)
    if is_raw:
        for (ds_id, col), fn in RAW_CONVERSIONS.items():
            if ds_id == dataset_id and col in df.columns:
                df[col] = fn(df[col])

    dt_col = _detect_datetime_col(df)
    t = _parse_datetime(df[dt_col])
    # Sort by datetime (raw data may not be sorted)
    df = df.iloc[t.argsort()].reset_index(drop=True)
    t = t.sort_values().reset_index(drop=True)

    groups = _resolve_groups(df, dataset_id, is_raw, user_cols, n_seq)
    if not groups:
        print("ERROR: no valid columns to plot")
        return

    if full_range:
        mask = slice(None)
        title_suffix = "full range"
    else:
        t_min = t.min()
        t_max = t_min + pd.Timedelta(weeks=n_weeks)
        mask = (t >= t_min) & (t < t_max)
        title_suffix = f"first {n_weeks} week(s)"

    for grp_name, cols in groups:
        out_path = _make_output_path(output, dataset_id, grp_name, is_raw)
        print(f"  [{grp_name}] {cols}")
        _plot_wide(df, dt_col, mask, cols, f"{dataset_id}/{grp_name}",
                   title_suffix, out_path)


def _plot_wide(df, dt_col, mask, cols, title_prefix, title_suffix, out_path):
    t = _parse_datetime(df[dt_col])
    fig, ax = plt.subplots(figsize=(14, 5))
    for c in cols:
        s = pd.to_numeric(df[c].loc[mask], errors="coerce")
        ax.plot(t.loc[mask], s.values, linewidth=0.5, label=c, alpha=0.85)

    ax.set_title(f"{title_prefix} — {len(cols)} sequences ({title_suffix})")
    ax.set_ylabel("Value")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    if len(cols) <= 10:
        ax.legend(fontsize=7, ncol=min(4, len(cols)))
    fig.tight_layout()

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot power time series from any CSV dataset")
    ap.add_argument("--input", required=True,
                    help="Path to CSV file (raw or processed)")
    ap.add_argument("--cols", default=None,
                    help="Comma-separated column names (overrides hardcoded groups)")
    ap.add_argument("--n", type=int, default=6,
                    help="Max sequences per group (default 6)")
    ap.add_argument("--weeks", type=int, default=2,
                    help="Number of weeks to show (default 2)")
    ap.add_argument("--full", action="store_true",
                    help="Plot full time range")
    ap.add_argument("--output", default=None,
                    help="Output path base (group name inserted before .png)")
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
