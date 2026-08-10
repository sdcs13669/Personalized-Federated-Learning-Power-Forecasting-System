#!/usr/bin/env python3
"""Clean script for eld_ind (ElectricityLoadDiagrams 2011-2014, UCI).

- 统一 timestep 1800s (30min), resample=mean (原生 15min)
- 异常检测: IQR (Q1−1.5×IQR, Q3+1.5×IQR) 逐列 → 置 NaN
- 物理边界: > 0 (功率值无上界)
- 填充策略: 根据清洗后缺失率与最大连续缺口决定 (≤6步插值, 大缺口保留 NaN)
- 向量化处理 370 列, 无本地特征 (§5.2.1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT  / "raw" / "eld_ind"
PROC = ROOT  / "processed"
PROC.mkdir(exist_ok=True)

DATASET_ID = "eld_ind"
TARGET_TIMESTEP = 1800  # 30 min
CATEGORY_ID = 1          # 变压器
IQR_MULTIPLIER_LABEL = 2.5
ROLLING_WINDOW = 336   # 7 days @ 30min
MAX_GAP_DROP = 48     # drop sequences with raw gap > 48 steps (24h)


def _max_consecutive_nan(arr: np.ndarray) -> int:
    """Max consecutive NaN after first valid (non-NaN) value.

    Leading NaN block (transformer not yet connected) is NOT counted as a gap.
    """
    is_nan = np.isnan(arr)
    valid_idx = np.where(~is_nan)[0]
    if len(valid_idx) == 0:
        return len(arr)
    start = valid_idx[0]
    is_nan_after = is_nan[start:]
    if not is_nan_after.any():
        return 0
    boundaries = np.diff(np.concatenate(([True], ~is_nan_after, [True])))
    runs = np.where(boundaries)[0]
    return (runs[1::2] - runs[::2]).max()


def load_raw() -> pd.DataFrame:
    csv_files = list(RAW.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {RAW}")
    df = pd.read_csv(csv_files[0], low_memory=False)
    df.columns = [c.strip().strip('"') for c in df.columns]
    ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    df = df.rename(columns={ts_col: "datetime"})
    return df


def clean() -> pd.DataFrame:
    df = load_raw()
    client_cols = [c for c in df.columns if c != "datetime"]
    print(f"Wide-format: {len(df)} rows, {len(client_cols)} clients")

    # Resample to 30min
    df = df.set_index("datetime")
    df = df[client_cols].resample(f"{TARGET_TIMESTEP}s").mean()
    print(f"After resample: {len(df)} rows")

    # ---- Step 0: drop sequences with raw gaps > 48 steps (24h) ----
    X = df.values.astype(np.float64)
    n_before = X.shape[1]
    keep_idx = []
    for j in range(X.shape[1]):
        col = X[:, j]
        is_bad = np.isnan(col) | (col == 0)
        if not is_bad.any():
            keep_idx.append(j)
            continue
        # Measure gap from first valid (non-NaN, >0), skip leading zeros
        valid_idx = np.where(~is_bad)[0]
        if len(valid_idx) == 0:
            continue
        start = valid_idx[0]
        is_bad_after = is_bad[start:]
        if not is_bad_after.any():
            keep_idx.append(j)
            continue
        boundaries = np.diff(np.concatenate(([True], ~is_bad_after, [True])))
        runs = np.where(boundaries)[0]
        max_run = (runs[1::2] - runs[::2]).max()
        if max_run <= MAX_GAP_DROP:
            keep_idx.append(j)
    n_dropped = n_before - len(keep_idx)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} sequences with raw max_gap > {MAX_GAP_DROP}")
    df = df.iloc[:, keep_idx]
    client_cols = [client_cols[i] for i in keep_idx]

    X = df.values.astype(np.float64)

    # ---- Step 0.5: trim leading zeros (transformer not yet connected) ----
    for j in range(X.shape[1]):
        col = X[:, j]
        valid = np.where(~np.isnan(col) & (col > 0))[0]
        if len(valid) > 0 and valid[0] > 0:
            X[:valid[0], j] = np.nan

    # ---- Step 1: outlier detection on diffs (labels: detect spikes) ----
    d = np.diff(X, axis=0, prepend=X[:1, :])  # signed first-order diffs
    q1 = np.nanpercentile(d, 25, axis=0)
    q3 = np.nanpercentile(d, 75, axis=0)
    iqr = q3 - q1
    lo = q1 - IQR_MULTIPLIER_LABEL * iqr
    hi = q3 + IQR_MULTIPLIER_LABEL * iqr
    outlier_mask = (d < lo) | (d > hi)
    n_outliers = outlier_mask.sum()
    n_seqs = X.shape[1]
    print(f"IQR flagged {n_outliers} outliers total "
          f"({n_outliers / n_seqs:.0f} avg per sequence, "
          f"{100 * n_outliers / outlier_mask.size:.2f}% of all values)")
    X[outlier_mask] = np.nan
    X = np.clip(X, 0.0, None)  # physical: > 0 only

    # ---- Step 2: statistics (gap measured from first valid, after trimming) ----
    missing_rate = np.isnan(X).mean()
    max_gaps = [_max_consecutive_nan(X[:, j]) for j in range(X.shape[1])]
    max_gap = max(max_gaps)
    print(f"After outlier marking: missing_rate={missing_rate:.4f}, "
          f"max_gap={max_gap} steps")

    # ---- Step 2.5: drop sequences with post-cleaning max_gap > 48 ----
    post_keep = [j for j, g in enumerate(max_gaps) if g <= MAX_GAP_DROP]
    n_post_dropped = X.shape[1] - len(post_keep)
    if n_post_dropped > 0:
        print(f"Dropped {n_post_dropped} sequences with post-cleaning "
              f"max_gap > {MAX_GAP_DROP}")
        X = X[:, post_keep]
        df = df.iloc[:, post_keep]
        client_cols = [client_cols[i] for i in post_keep]

    # ---- Step 2.6: cubic spline interpolation (per column, interior only) ----
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.isnan(col).any():
            s = pd.Series(col)
            s = s.interpolate(method="cubic", limit_area="inside")
            s = s.clip(0.0, None)  # physical: power > 0
            X[:, j] = s.values

    # ---- Step 3: public features ----
    df_out = pd.DataFrame(X, index=df.index, columns=df.columns)
    df_out = df_out.reset_index()
    t = pd.to_datetime(df_out["datetime"])
    hour = t.dt.hour + t.dt.minute / 60.0
    dow = t.dt.dayofweek
    month = t.dt.month
    df_out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df_out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df_out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df_out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df_out["is_weekend"] = (dow >= 5).astype(int)
    df_out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df_out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    df_out["category_id"] = CATEGORY_ID

    keep = (["datetime"] + client_cols +
            ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
             "month_sin", "month_cos", "category_id"])
    return df_out[keep]


def main() -> None:
    df = clean()
    out = PROC / f"{DATASET_ID}.csv"
    df.to_csv(out, index=False)
    client_cols = [c for c in df.columns if c.startswith("MT_")]
    sub = df[client_cols]
    n_public = len(df.columns) - len(client_cols)
    print(f"Wrote {out} ({len(df)} rows, {len(client_cols)} MT + {n_public} public = {len(df.columns)} cols)")
    print(f"Final missing_rate={sub.isna().mean().mean():.4f} "
          f"range=[{sub.min().min():.2f}, {sub.max().max():.2f}]")


if __name__ == "__main__":
    main()
