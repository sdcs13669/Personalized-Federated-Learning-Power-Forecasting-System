#!/usr/bin/env python3
"""Analyze processed data: per-sequence stats (missing rate, max gap, distribution).

Usage:
  python data/scripts/analysis/processed/analyze_processed.py <dataset_id>
  python data/scripts/analysis/processed/analyze_processed.py eld_ind
  python data/scripts/analysis/processed/analyze_processed.py --all

Output: data/scripts/analysis/processed/<dataset_id>_profile.txt
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parent

SKIP_COLS = {
    "datetime", "LCLid", "tariff", "Load_Type", "load_type",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "month_sin", "month_cos", "category_id",
}


def _max_consecutive_nan(arr: np.ndarray) -> int:
    """Max consecutive NaN from first valid (non-NaN, >0) position."""
    valid_idx = np.where(~np.isnan(arr) & (arr > 0))[0]
    if len(valid_idx) == 0:
        return len(arr)
    start = valid_idx[0]
    is_nan = np.isnan(arr[start:])
    if not is_nan.any():
        return 0
    boundaries = np.diff(np.concatenate(([True], ~is_nan, [True])))
    runs = np.where(boundaries)[0]
    return (runs[1::2] - runs[::2]).max()


def _missing_rate(arr: np.ndarray) -> float:
    """NaN fraction from first valid position."""
    valid_idx = np.where(~np.isnan(arr) & (arr > 0))[0]
    if len(valid_idx) == 0:
        return 1.0
    seg = arr[valid_idx[0]:]
    return np.isnan(seg).mean()


def _per_sequence_stats(X: np.ndarray, col_names: list) -> dict:
    """Compute per-sequence stats for wide-format data."""
    n_seqs = X.shape[1]
    stats = {
        "effective_len": [], "missing_rate": [], "max_gap": [],
        "pmin": [], "p5": [], "q1": [], "median": [], "q3": [], "p95": [], "pmax": [],
    }

    for j in range(n_seqs):
        col = X[:, j]
        valid = np.where(~np.isnan(col) & (col > 0))[0]
        if len(valid) == 0:
            stats["effective_len"].append(0)
            stats["missing_rate"].append(1.0)
            stats["max_gap"].append(len(col))
            for k in ["pmin", "p5", "q1", "median", "q3", "p95", "pmax"]:
                stats[k].append(np.nan)
            continue

        start = valid[0]
        seg = col[start:]
        seg_valid = seg[~np.isnan(seg)]

        stats["effective_len"].append(len(seg))
        stats["missing_rate"].append(np.isnan(seg).mean())
        stats["max_gap"].append(_max_consecutive_nan(col))
        stats["pmin"].append(np.min(seg_valid))
        stats["p5"].append(np.percentile(seg_valid, 5))
        stats["q1"].append(np.percentile(seg_valid, 25))
        stats["median"].append(np.median(seg_valid))
        stats["q3"].append(np.percentile(seg_valid, 75))
        stats["p95"].append(np.percentile(seg_valid, 95))
        stats["pmax"].append(np.max(seg_valid))

    return stats


def _agg_stats(stats: dict, key: str, fmt: str = ".1f") -> str:
    vals = np.array([v for v in stats[key] if not np.isnan(v)])
    if len(vals) == 0:
        return "N/A"
    return (f"min={vals.min():{fmt}}  q1={np.percentile(vals, 25):{fmt}}  "
            f"median={np.median(vals):{fmt}}  q3={np.percentile(vals, 75):{fmt}}  "
            f"max={vals.max():{fmt}}")


def analyze_eld_ind() -> str:
    """Wide-format: MT_ columns + public features."""
    path = PROC / "eld_ind.csv"
    if not path.exists():
        return f"ERROR: {path} not found"
    df = pd.read_csv(path, low_memory=False)
    mt_cols = [c for c in df.columns if c.startswith("MT_")]
    X = df[mt_cols].values.astype(np.float64)
    n_seqs = X.shape[1]
    stats = _per_sequence_stats(X, mt_cols)

    lines = []
    lines.append("=" * 70)
    lines.append("eld_ind — Processed Data Profile")
    lines.append("=" * 70)
    lines.append(f"  Total rows:         {len(df)}")
    lines.append(f"  Total sequences:    {n_seqs}")
    active = sum(1 for x in stats["effective_len"] if x > 0)
    dead = sum(1 for x in stats["effective_len"] if x == 0)
    lines.append(f"  Active sequences:   {active}")
    lines.append(f"  Dead sequences:     {dead}")
    lines.append("")

    lines.append("--- Per-sequence stats (aggregated across all sequences) ---")
    lines.append(f"  effective_length (30min steps): {_agg_stats(stats, 'effective_len')}")
    lines.append(f"  missing_rate:                   {_agg_stats(stats, 'missing_rate', '.4f')}")
    lines.append(f"  max_gap (consecutive NaN):      {_agg_stats(stats, 'max_gap')}")
    lines.append(f"  min:      {_agg_stats(stats, 'pmin')}")
    lines.append(f"  p5:       {_agg_stats(stats, 'p5')}")
    lines.append(f"  q1:       {_agg_stats(stats, 'q1')}")
    lines.append(f"  median:   {_agg_stats(stats, 'median')}")
    lines.append(f"  q3:       {_agg_stats(stats, 'q3')}")
    lines.append(f"  p95:      {_agg_stats(stats, 'p95')}")
    lines.append(f"  max:      {_agg_stats(stats, 'pmax')}")
    lines.append("")

    # Detail: top 20 columns
    lines.append("--- Per-sequence detail (first 20 columns) ---")
    header = f"{'Col':<12} {'EffLen':>8} {'MissRate':>10} {'MaxGap':>8} {'Min':>10} {'Median':>10} {'Max':>10}"
    lines.append(header)
    for j in range(min(20, n_seqs)):
        e = stats["effective_len"][j]
        if e == 0:
            lines.append(f"  {mt_cols[j]:<10} {'DEAD':>8}")
        else:
            lines.append(f"  {mt_cols[j]:<10} {e:>8} {stats['missing_rate'][j]:>10.4f} "
                         f"{stats['max_gap'][j]:>8} {stats['pmin'][j]:>10.1f} "
                         f"{stats['median'][j]:>10.1f} {stats['pmax'][j]:>10.1f}")
    lines.append("")
    return "\n".join(lines)


def analyze_lcl_res() -> str:
    """Long-format: per-LCLid KWH."""
    path = PROC / "lcl_res.csv"
    if not path.exists():
        return f"ERROR: {path} not found"

    # Stream read — file can be very large
    user_stats = []
    chunk_iter = pd.read_csv(path, chunksize=500_000)
    for chunk in chunk_iter:
        for lid, g in chunk.groupby("LCLid", sort=False):
            g = g.sort_values("datetime").reset_index(drop=True)
            kwh = pd.to_numeric(g["KWH"], errors="coerce").values.astype(np.float64)
            valid = np.where(~np.isnan(kwh) & (kwh > 0))[0]
            if len(valid) == 0:
                user_stats.append({
                    "lid": lid, "eff_len": 0, "missing_rate": 1.0,
                    "max_gap": len(kwh),
                    "min": np.nan, "q1": np.nan, "median": np.nan,
                    "q3": np.nan, "max": np.nan,
                })
                continue
            start = valid[0]
            seg = kwh[start:]
            seg_valid = seg[~np.isnan(seg)]
            user_stats.append({
                "lid": lid, "eff_len": len(seg),
                "missing_rate": np.isnan(seg).mean(),
                "max_gap": _max_consecutive_nan(kwh),
                "min": np.min(seg_valid),
                "q1": np.percentile(seg_valid, 25),
                "median": np.median(seg_valid),
                "q3": np.percentile(seg_valid, 75),
                "max": np.max(seg_valid),
            })

    n_users = len(user_stats)
    active = sum(1 for u in user_stats if u["eff_len"] > 0)

    lines = []
    lines.append("=" * 70)
    lines.append("lcl_res — Processed Data Profile")
    lines.append("=" * 70)
    lines.append(f"  Unique LCLid:     {n_users}")
    lines.append(f"  Active users:     {active}")
    lines.append(f"  Dead users:       {n_users - active}")
    lines.append("")

    def _agg(key, fmt=".3f"):
        vals = np.array([u[key] for u in user_stats if not np.isnan(u[key])])
        if len(vals) == 0:
            return "N/A"
        return (f"min={vals.min():{fmt}}  q1={np.percentile(vals, 25):{fmt}}  "
                f"median={np.median(vals):{fmt}}  q3={np.percentile(vals, 75):{fmt}}  "
                f"max={vals.max():{fmt}}")

    lines.append("--- Per-user stats (aggregated) ---")
    lines.append(f"  effective_length (30min steps): {_agg('eff_len', '.0f')}")
    lines.append(f"  missing_rate:                   {_agg('missing_rate', '.4f')}")
    lines.append(f"  max_gap (consecutive NaN):      {_agg('max_gap', '.0f')}")
    lines.append(f"  min (kWh):     {_agg('min')}")
    lines.append(f"  q1 (kWh):      {_agg('q1')}")
    lines.append(f"  median (kWh):  {_agg('median')}")
    lines.append(f"  q3 (kWh):      {_agg('q3')}")
    lines.append(f"  max (kWh):     {_agg('max')}")
    lines.append("")

    lines.append("--- Per-user detail (first 20 users) ---")
    header = f"{'LCLid':<18} {'EffLen':>8} {'MissRate':>10} {'MaxGap':>8} {'Min':>10} {'Median':>10} {'Max':>10}"
    lines.append(header)
    for u in user_stats[:20]:
        if u["eff_len"] == 0:
            lines.append(f"  {u['lid']:<16} {'DEAD':>8}")
        else:
            lines.append(f"  {u['lid']:<16} {u['eff_len']:>8} {u['missing_rate']:>10.4f} "
                         f"{u['max_gap']:>8} {u['min']:>10.3f} "
                         f"{u['median']:>10.3f} {u['max']:>10.3f}")
    lines.append("")
    return "\n".join(lines)


def analyze_steel_ind() -> str:
    """Single plant: target + features."""
    path = PROC / "steel_ind.csv"
    if not path.exists():
        return f"ERROR: {path} not found"
    df = pd.read_csv(path)

    data_cols = [c for c in df.columns if c not in SKIP_COLS]
    lines = []
    lines.append("=" * 70)
    lines.append("steel_ind — Processed Data Profile")
    lines.append("=" * 70)
    lines.append(f"  Total rows: {len(df)}")
    lines.append("")

    lines.append("--- Per-column stats (from first valid) ---")
    header = f"{'Column':<28} {'EffLen':>8} {'MissRate':>10} {'MaxGap':>8} {'Min':>12} {'Median':>12} {'Max':>12}"
    lines.append(header)
    for c in data_cols:
        vals = pd.to_numeric(df[c], errors="coerce").values.astype(np.float64)
        valid = np.where(~np.isnan(vals) & (vals > 0))[0]
        if len(valid) == 0:
            lines.append(f"  {c:<26} {'DEAD':>8}")
            continue
        start = valid[0]
        seg = vals[start:]
        seg_valid = seg[~np.isnan(seg)]
        mr = np.isnan(seg).mean()
        mg = _max_consecutive_nan(vals)
        lines.append(f"  {c:<26} {len(seg):>8} {mr:>10.4f} {mg:>8} "
                     f"{np.min(seg_valid):>12.4f} {np.median(seg_valid):>12.4f} "
                     f"{np.max(seg_valid):>12.4f}")
    lines.append("")
    return "\n".join(lines)


def analyze_tetouan_city() -> str:
    """3 zones + 5 weather features."""
    path = PROC / "tetouan_city.csv"
    if not path.exists():
        return f"ERROR: {path} not found"
    df = pd.read_csv(path)

    data_cols = [c for c in df.columns if c not in SKIP_COLS]
    lines = []
    lines.append("=" * 70)
    lines.append("tetouan_city — Processed Data Profile")
    lines.append("=" * 70)
    lines.append(f"  Total rows: {len(df)}")
    lines.append("")

    lines.append("--- Per-column stats (from first valid) ---")
    header = f"{'Column':<26} {'EffLen':>8} {'MissRate':>10} {'MaxGap':>8} {'Min':>12} {'Median':>12} {'Max':>12}"
    lines.append(header)
    for c in data_cols:
        vals = pd.to_numeric(df[c], errors="coerce").values.astype(np.float64)
        valid = np.where(~np.isnan(vals) & (vals > 0))[0]
        if len(valid) == 0:
            lines.append(f"  {c:<24} {'DEAD':>8}")
            continue
        start = valid[0]
        seg = vals[start:]
        seg_valid = seg[~np.isnan(seg)]
        mr = np.isnan(seg).mean()
        mg = _max_consecutive_nan(vals)
        lines.append(f"  {c:<24} {len(seg):>8} {mr:>10.4f} {mg:>8} "
                     f"{np.min(seg_valid):>12.4f} {np.median(seg_valid):>12.4f} "
                     f"{np.max(seg_valid):>12.4f}")
    lines.append("")
    return "\n".join(lines)


def analyze_household_res() -> str:
    """Single household."""
    path = PROC / "household_res.csv"
    if not path.exists():
        return f"ERROR: {path} not found"
    df = pd.read_csv(path)

    data_cols = [c for c in df.columns if c not in SKIP_COLS]
    lines = []
    lines.append("=" * 70)
    lines.append("household_res — Processed Data Profile")
    lines.append("=" * 70)
    lines.append(f"  Total rows: {len(df)}")
    lines.append("")

    lines.append("--- Per-column stats (from first valid) ---")
    header = f"{'Column':<28} {'EffLen':>8} {'MissRate':>10} {'MaxGap':>8} {'Min':>12} {'Median':>12} {'Max':>12}"
    lines.append(header)
    for c in data_cols:
        vals = pd.to_numeric(df[c], errors="coerce").values.astype(np.float64)
        valid = np.where(~np.isnan(vals) & (vals > 0))[0]
        if len(valid) == 0:
            lines.append(f"  {c:<26} {'DEAD':>8}")
            continue
        start = valid[0]
        seg = vals[start:]
        seg_valid = seg[~np.isnan(seg)]
        mr = np.isnan(seg).mean()
        mg = _max_consecutive_nan(vals)
        lines.append(f"  {c:<26} {len(seg):>8} {mr:>10.4f} {mg:>8} "
                     f"{np.min(seg_valid):>12.4f} {np.median(seg_valid):>12.4f} "
                     f"{np.max(seg_valid):>12.4f}")
    lines.append("")
    return "\n".join(lines)


def main():
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["--all"]
    if "--all" in datasets:
        datasets = ["eld_ind", "lcl_res", "steel_ind", "tetouan_city", "household_res"]

    analyzers = {
        "eld_ind": analyze_eld_ind,
        "lcl_res": analyze_lcl_res,
        "steel_ind": analyze_steel_ind,
        "tetouan_city": analyze_tetouan_city,
        "household_res": analyze_household_res,
    }

    for ds in datasets:
        if ds not in analyzers:
            print(f"Unknown dataset: {ds}")
            continue
        print(f"Analyzing {ds} (processed)...")
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
