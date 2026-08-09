#!/usr/bin/env python3
"""Analyze raw data: resample, extract per-sequence stats from first non-zero.

Usage:
  python data/scripts/analysis/analyze_raw.py <dataset_id>
  python data/scripts/analysis/analyze_raw.py eld_ind
  python data/scripts/analysis/analyze_raw.py --all

Output: data/scripts/analysis/<dataset_id>_profile.txt
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent
TARGET_TIMESTEP = 1800  # 30 min


def _max_consecutive_zeros(arr: np.ndarray) -> int:
    """Max consecutive zeros or NaN in a 1D array (before first valid)."""
    is_bad = np.isnan(arr) | (arr == 0)
    if not is_bad.any():
        return 0
    boundaries = np.diff(np.concatenate(([True], ~is_bad, [True])))
    runs = np.where(boundaries)[0]
    return (runs[1::2] - runs[::2]).max()


def analyze_eld_ind() -> str:
    """Wide-format: 370 MT columns, all targets."""
    csv_files = list((RAW / "eld_ind").glob("*.csv"))
    if not csv_files:
        return "ERROR: no CSV found"
    df = pd.read_csv(csv_files[0], low_memory=False)
    df.columns = [c.strip().strip('"') for c in df.columns]
    ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)
    df = df.rename(columns={ts_col: "datetime"})

    mt_cols = [c for c in df.columns if c != "datetime"]
    total_rows = len(df)

    # Resample
    df = df.set_index("datetime")
    df = df[mt_cols].resample(f"{TARGET_TIMESTEP}s").mean()
    resampled_rows = len(df)

    X = df.values.astype(np.float64)
    n_seqs = X.shape[1]

    stats = {
        "effective_len": [], "first_valid_idx": [],
        "pmin": [], "p5": [], "q1": [], "median": [], "q3": [], "p95": [], "pmax": [],
        "max_gap": [],
    }

    for j in range(n_seqs):
        col = X[:, j]
        valid = np.where(~np.isnan(col) & (col > 0))[0]
        if len(valid) == 0:
            stats["effective_len"].append(0)
            stats["first_valid_idx"].append(-1)
            for k in ["pmin", "p5", "q1", "median", "q3", "p95", "pmax", "max_gap"]:
                stats[k].append(np.nan)
            continue

        start = valid[0]
        seg = col[start:]
        seg_valid = seg[~np.isnan(seg)]

        stats["effective_len"].append(len(seg))
        stats["first_valid_idx"].append(start)
        stats["pmin"].append(np.min(seg_valid))
        stats["p5"].append(np.percentile(seg_valid, 5))
        stats["q1"].append(np.percentile(seg_valid, 25))
        stats["median"].append(np.median(seg_valid))
        stats["q3"].append(np.percentile(seg_valid, 75))
        stats["p95"].append(np.percentile(seg_valid, 95))
        stats["pmax"].append(np.max(seg_valid))
        stats["max_gap"].append(_max_consecutive_zeros(seg))

    # Build report
    lines = []
    lines.append("=" * 70)
    lines.append("eld_ind — Raw Data Profile")
    lines.append("=" * 70)
    lines.append(f"  Source rows (15min):  {total_rows}")
    lines.append(f"  After resample (30min): {resampled_rows}")
    lines.append(f"  Total sequences (MT columns): {n_seqs}")
    lines.append(f"  Active sequences (>0 rows):   {sum(1 for x in stats['effective_len'] if x > 0)}")
    lines.append(f"  Dead sequences (all zero):    {sum(1 for x in stats['effective_len'] if x == 0)}")
    lines.append("")

    # Aggregate stats across sequences
    def _agg(name):
        vals = np.array([v for v in stats[name] if not np.isnan(v)])
        if len(vals) == 0:
            return "N/A"
        return (f"min={vals.min():.1f}  q1={np.percentile(vals,25):.1f}  "
                f"median={np.median(vals):.1f}  q3={np.percentile(vals,75):.1f}  "
                f"max={vals.max():.1f}")

    lines.append("--- Per-sequence stats (aggregated across all sequences) ---")
    lines.append(f"  effective_length (30min steps): {_agg('effective_len')}")
    lines.append(f"  first_valid_idx (30min steps):  {_agg('first_valid_idx')}")
    lines.append(f"  min (kW):     {_agg('pmin')}")
    lines.append(f"  p5 (kW):      {_agg('p5')}")
    lines.append(f"  q1 (kW):      {_agg('q1')}")
    lines.append(f"  median (kW):  {_agg('median')}")
    lines.append(f"  q3 (kW):      {_agg('q3')}")
    lines.append(f"  p95 (kW):     {_agg('p95')}")
    lines.append(f"  max (kW):     {_agg('pmax')}")
    lines.append(f"  max_gap (30min steps, zeros or NaN): {_agg('max_gap')}")
    lines.append("")

    # Per-sequence detail (top 20 + summary)
    lines.append("--- Per-sequence detail (first 20 columns) ---")
    lines.append(f"{'Col':<12} {'EffLen':>8} {'FirstIdx':>9} {'Min':>10} {'Median':>10} {'Max':>10} {'MaxGap':>8}")
    for j in range(min(20, n_seqs)):
        name = mt_cols[j]
        e = stats["effective_len"][j]
        fi = stats["first_valid_idx"][j]
        if e == 0:
            lines.append(f"  {name:<10} {'DEAD':>8}")
        else:
            lines.append(f"  {name:<10} {e:>8} {fi:>9} "
                         f"{stats['pmin'][j]:>10.1f} {stats['median'][j]:>10.1f} "
                         f"{stats['pmax'][j]:>10.1f} {stats['max_gap'][j]:>8}")
    lines.append("")

    return "\n".join(lines)


def analyze_lcl_res() -> str:
    """Long-format: per-LCLid sequences."""
    csv_files = list((RAW / "lcl_res").glob("*.csv"))
    if not csv_files:
        return "ERROR: no CSV found"
    df = pd.read_csv(csv_files[0])
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "KWH/hh (per half hour)": "KWH",
        "DateTime": "datetime",
    })
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["KWH"] = pd.to_numeric(df["KWH"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    total_rows = len(df)
    n_users = df["LCLid"].nunique()

    # Per-user stats
    user_stats = []
    for lid, g in df.groupby("LCLid", sort=True):
        g = g.sort_values("datetime").reset_index(drop=True)
        kwh = g["KWH"].values.astype(np.float64)
        valid = np.where(~np.isnan(kwh) & (kwh > 0))[0]
        if len(valid) == 0:
            user_stats.append({
                "lid": lid, "eff_len": 0, "first_idx": -1,
                "min": np.nan, "median": np.nan, "max": np.nan,
                "q1": np.nan, "q3": np.nan, "max_gap": np.nan,
            })
            continue
        start = valid[0]
        seg = kwh[start:]
        seg_valid = seg[~np.isnan(seg)]
        user_stats.append({
            "lid": lid, "eff_len": len(seg), "first_idx": start,
            "min": np.min(seg_valid), "median": np.median(seg_valid),
            "max": np.max(seg_valid),
            "q1": np.percentile(seg_valid, 25),
            "q3": np.percentile(seg_valid, 75),
            "max_gap": _max_consecutive_zeros(seg),
        })

    lines = []
    lines.append("=" * 70)
    lines.append("lcl_res — Raw Data Profile")
    lines.append("=" * 70)
    lines.append(f"  Source rows:        {total_rows}")
    lines.append(f"  Unique LCLid:       {n_users}")
    lines.append(f"  Active users (>0):  {sum(1 for u in user_stats if u['eff_len'] > 0)}")
    lines.append(f"  Dead users (all 0): {sum(1 for u in user_stats if u['eff_len'] == 0)}")
    lines.append("")

    def _agg(key):
        vals = np.array([u[key] for u in user_stats if not np.isnan(u[key])])
        if len(vals) == 0:
            return "N/A"
        return (f"min={vals.min():.3f}  q1={np.percentile(vals,25):.3f}  "
                f"median={np.median(vals):.3f}  q3={np.percentile(vals,75):.3f}  "
                f"max={vals.max():.3f}")

    lines.append("--- Per-user stats (aggregated across all users) ---")
    lines.append(f"  effective_length (30min steps): {_agg('eff_len')}")
    lines.append(f"  first_valid_idx (30min steps):  {_agg('first_idx')}")
    lines.append(f"  min (kWh):     {_agg('min')}")
    lines.append(f"  median (kWh):  {_agg('median')}")
    lines.append(f"  max (kWh):     {_agg('max')}")
    lines.append(f"  q1 (kWh):      {_agg('q1')}")
    lines.append(f"  q3 (kWh):      {_agg('q3')}")
    lines.append(f"  max_gap (30min steps): {_agg('max_gap')}")
    lines.append("")

    # Top 20 users
    lines.append("--- Per-user detail (first 20 users) ---")
    lines.append(f"{'LCLid':<18} {'EffLen':>8} {'FirstIdx':>9} {'Min':>10} {'Median':>10} {'Max':>10} {'MaxGap':>8}")
    for u in user_stats[:20]:
        if u["eff_len"] == 0:
            lines.append(f"  {u['lid']:<16} {'DEAD':>8}")
        else:
            lines.append(f"  {u['lid']:<16} {u['eff_len']:>8} {u['first_idx']:>9} "
                         f"{u['min']:>10.3f} {u['median']:>10.3f} "
                         f"{u['max']:>10.3f} {u['max_gap']:>8}")
    lines.append("")

    return "\n".join(lines)


def analyze_steel_ind() -> str:
    """Single industrial plant."""
    csv_files = list((RAW / "steel_ind").glob("*.csv"))
    if not csv_files:
        return "ERROR: no CSV found"
    df = pd.read_csv(csv_files[0])
    df["datetime"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M", errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    total_rows = len(df)

    target = "Usage_kWh"
    s = pd.to_numeric(df[target], errors="coerce").values.astype(np.float64)

    # Resample
    df = df.set_index("datetime")
    s_30m = df[target].resample(f"{TARGET_TIMESTEP}s").mean()
    resampled_rows = len(s_30m)
    vals = s_30m.values.astype(np.float64)

    valid = np.where(~np.isnan(vals) & (vals > 0))[0]
    if len(valid) == 0:
        return "steel_ind: all zero/NaN"

    start = valid[0]
    seg = vals[start:]
    seg_valid = seg[~np.isnan(seg)]

    lines = []
    lines.append("=" * 70)
    lines.append("steel_ind — Raw Data Profile")
    lines.append("=" * 70)
    lines.append(f"  Source rows (15min):  {total_rows}")
    lines.append(f"  After resample (30min): {resampled_rows}")
    lines.append(f"  First valid idx:  {start}")
    lines.append(f"  Effective length: {len(seg)}")
    lines.append(f"  min:      {np.min(seg_valid):.3f}")
    lines.append(f"  p5:       {np.percentile(seg_valid, 5):.3f}")
    lines.append(f"  q1:       {np.percentile(seg_valid, 25):.3f}")
    lines.append(f"  median:   {np.median(seg_valid):.3f}")
    lines.append(f"  q3:       {np.percentile(seg_valid, 75):.3f}")
    lines.append(f"  p95:      {np.percentile(seg_valid, 95):.3f}")
    lines.append(f"  max:      {np.max(seg_valid):.3f}")
    lines.append(f"  max_gap (30min steps, zeros or NaN): {_max_consecutive_zeros(seg)}")
    lines.append("")

    # Also analyze local features
    feat_cols = [
        "Lagging_Current_Reactive.Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh",
        "CO2(tCO2)",
        "Lagging_Current_Power_Factor",
        "Leading_Current_Power_Factor",
    ]
    df_raw = pd.read_csv(csv_files[0])
    lines.append("--- Local features (raw, before resample) ---")
    for c in feat_cols:
        if c in df_raw.columns:
            fs = pd.to_numeric(df_raw[c], errors="coerce").dropna()
            if len(fs) > 0:
                lines.append(f"  {c}: min={fs.min():.4f}  q1={fs.quantile(0.25):.4f}  "
                             f"median={fs.median():.4f}  q3={fs.quantile(0.75):.4f}  "
                             f"max={fs.max():.4f}")
    lines.append("")

    return "\n".join(lines)


def analyze_tetouan_city() -> str:
    """3 zones + 5 weather features."""
    csv_files = list((RAW / "tetouan_city").glob("*.csv"))
    if not csv_files:
        return "ERROR: no CSV found"
    df = pd.read_csv(csv_files[0])
    df["datetime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    total_rows = len(df)

    zone_cols = {
        "Zone 1 Power Consumption": "Zone 1",
        "Zone 2  Power Consumption": "Zone 2",
        "Zone 3  Power Consumption": "Zone 3",
    }

    # Resample
    df = df.set_index("datetime")
    num_cols = list(zone_cols.keys()) + ["Temperature", "Humidity", "Wind Speed",
                                          "general diffuse flows", "diffuse flows"]
    df_30m = df[[c for c in num_cols if c in df.columns]].resample(
        f"{TARGET_TIMESTEP}s").mean()
    resampled_rows = len(df_30m)

    lines = []
    lines.append("=" * 70)
    lines.append("tetouan_city — Raw Data Profile")
    lines.append("=" * 70)
    lines.append(f"  Source rows (10min):  {total_rows}")
    lines.append(f"  After resample (30min): {resampled_rows}")
    lines.append("")

    # Zone targets
    lines.append("--- Zone power (targets, after resample, from first non-zero) ---")
    for col, label in zone_cols.items():
        if col not in df_30m.columns:
            lines.append(f"  {label}: column not found")
            continue
        vals = df_30m[col].values.astype(np.float64)
        valid = np.where(~np.isnan(vals) & (vals > 0))[0]
        if len(valid) == 0:
            lines.append(f"  {label}: all zero/NaN")
            continue
        start = valid[0]
        seg = vals[start:]
        seg_valid = seg[~np.isnan(seg)]
        lines.append(f"  {label}: first_valid={start}  eff_len={len(seg)}  "
                     f"min={np.min(seg_valid):.1f}  q1={np.percentile(seg_valid,25):.1f}  "
                     f"median={np.median(seg_valid):.1f}  q3={np.percentile(seg_valid,75):.1f}  "
                     f"max={np.max(seg_valid):.1f}  max_gap={_max_consecutive_zeros(seg)}")
    lines.append("")

    # Weather features
    lines.append("--- Weather features (after resample, from first valid) ---")
    weather_cols = ["Temperature", "Humidity", "Wind Speed",
                    "general diffuse flows", "diffuse flows"]
    for col in weather_cols:
        if col not in df_30m.columns:
            continue
        vals = df_30m[col].values.astype(np.float64)
        valid = np.where(~np.isnan(vals))[0]
        if len(valid) == 0:
            continue
        start = valid[0]
        seg = vals[start:]
        seg_valid = seg[~np.isnan(seg)]
        lines.append(f"  {col}: first_valid={start}  eff_len={len(seg)}  "
                     f"min={np.min(seg_valid):.2f}  q1={np.percentile(seg_valid,25):.2f}  "
                     f"median={np.median(seg_valid):.2f}  q3={np.percentile(seg_valid,75):.2f}  "
                     f"max={np.max(seg_valid):.2f}  max_gap={_max_consecutive_zeros(seg)}")
    lines.append("")

    return "\n".join(lines)


def main():
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["--all"]
    if "--all" in datasets:
        datasets = ["eld_ind", "lcl_res", "steel_ind", "tetouan_city"]

    analyzers = {
        "eld_ind": analyze_eld_ind,
        "lcl_res": analyze_lcl_res,
        "steel_ind": analyze_steel_ind,
        "tetouan_city": analyze_tetouan_city,
    }

    for ds in datasets:
        if ds not in analyzers:
            print(f"Unknown dataset: {ds}")
            continue
        print(f"Analyzing {ds}...")
        try:
            report = analyzers[ds]()
        except Exception as e:
            report = f"ERROR analyzing {ds}: {e}"
            import traceback
            traceback.print_exc()

        out_path = OUT_DIR / f"{ds}_profile.txt"
        out_path.write_text(report, encoding="utf-8")
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()
