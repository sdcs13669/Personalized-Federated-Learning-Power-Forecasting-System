#!/usr/bin/env python3
"""Clean script for lcl_res (Low Carbon London smart meter, London Datastore).

- 宽表输入: datetime + MAC000002, MAC000003, ... (已 pivot, 每列一户)
- 原生即 30min, 无需重采样
- 异常检测: diff IQR (系数 2.5) 逐户独立, 不跨户借信息 (§8 防泄漏)
- 物理边界: KWH > 0
- 填充: 三次样条 (interior only)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT /  "raw" / "lcl_res"
PROC = ROOT /  "processed"
PROC.mkdir(exist_ok=True)

DATASET_ID = "lcl_res"
CATEGORY_ID = 0          # 居民
IQR_MULTIPLIER_LABEL = 2.5
MAX_GAP_DROP = 48        # drop users with gap > 48 steps (24h)


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


def load_raw() -> pd.DataFrame:
    csv_files = list(RAW.glob("lcl_res.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No lcl_res.csv in {RAW}")
    df = pd.read_csv(csv_files[0], low_memory=False)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df


def add_public_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df["datetime"]
    hour = t.dt.hour + t.dt.minute / 60.0
    dow = t.dt.dayofweek
    month = t.dt.month
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    df["category_id"] = CATEGORY_ID
    return df


def clean() -> pd.DataFrame:
    df = load_raw()
    user_cols = [c for c in df.columns if c.startswith("MAC")]
    n_users = len(user_cols)
    print(f"Wide-format: {len(df)} rows, {n_users} users")

    df = df.set_index("datetime")

    # ---- Step 0: drop users with raw gaps > 48 steps ----
    X = df[user_cols].values.astype(np.float64)
    keep_idx = []
    for j in range(X.shape[1]):
        col = X[:, j]
        is_bad = np.isnan(col) | (col == 0)
        if not is_bad.any():
            keep_idx.append(j)
            continue
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
    n_dropped = n_users - len(keep_idx)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} users with raw max_gap > {MAX_GAP_DROP}")
    X = X[:, keep_idx]
    user_cols = [user_cols[i] for i in keep_idx]

    # ---- Step 0.5: trim leading zeros per user ----
    for j in range(X.shape[1]):
        col = X[:, j]
        valid = np.where(~np.isnan(col) & (col > 0))[0]
        if len(valid) > 0 and valid[0] > 0:
            X[:valid[0], j] = np.nan

    # ---- Step 1: diff IQR per user ----
    d = np.diff(X, axis=0, prepend=X[:1, :])
    q1 = np.nanpercentile(d, 25, axis=0)
    q3 = np.nanpercentile(d, 75, axis=0)
    iqr = q3 - q1
    lo = q1 - IQR_MULTIPLIER_LABEL * iqr
    hi = q3 + IQR_MULTIPLIER_LABEL * iqr
    outlier_mask = (d < lo) | (d > hi)
    n_outliers = outlier_mask.sum()
    n_seqs = X.shape[1]
    print(f"IQR flagged {n_outliers} outliers total "
          f"({n_outliers / n_seqs:.0f} avg per user, "
          f"{100 * n_outliers / outlier_mask.size:.2f}% of all values)")
    X[outlier_mask] = np.nan
    X = np.clip(X, 0.0, None)

    # ---- Step 2: post-cleaning gap filter ----
    max_gaps = [_max_consecutive_nan(X[:, j]) for j in range(X.shape[1])]
    post_keep = [j for j, g in enumerate(max_gaps) if g <= MAX_GAP_DROP]
    n_post_dropped = X.shape[1] - len(post_keep)
    if n_post_dropped > 0:
        print(f"Dropped {n_post_dropped} users with post-cleaning "
              f"max_gap > {MAX_GAP_DROP}")
        X = X[:, post_keep]
        user_cols = [user_cols[i] for i in post_keep]

    # ---- Step 3: cubic spline interpolation + re-clip ----
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.isnan(col).any():
            s = pd.Series(col)
            s = s.interpolate(method="cubic", limit_area="inside")
            X[:, j] = s.values
    X = np.clip(X, 0.0, None)

    # ---- Step 4: build output ----
    df_out = pd.DataFrame(X, index=df.index, columns=user_cols)
    df_out = df_out.reset_index()
    df_out = add_public_features(df_out)

    keep = (["datetime"] + user_cols +
            ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
             "month_sin", "month_cos", "category_id"])
    return df_out[keep]


def main() -> None:
    df = clean()
    out = PROC / f"{DATASET_ID}.csv"
    df.to_csv(out, index=False)
    user_cols = [c for c in df.columns if c.startswith("MAC")]
    sub = df[user_cols]
    n_public = len(df.columns) - len(user_cols)
    print(f"Wrote {out} ({len(df)} rows, "
          f"{len(user_cols)} users + {n_public} public = {len(df.columns)} cols)")
    print(f"Final KWH missing_rate={sub.isna().mean().mean():.4f} "
          f"range=[{sub.min().min():.3f}, {sub.max().max():.3f}]")


if __name__ == "__main__":
    main()
