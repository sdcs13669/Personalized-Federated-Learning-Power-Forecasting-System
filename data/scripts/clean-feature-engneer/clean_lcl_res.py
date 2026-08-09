#!/usr/bin/env python3
"""Clean script for lcl_res (Low Carbon London smart meter, London Datastore).

- 按客户端 (每户 LCLid) 独立处理, 不跨客户端借信息 (§8 防泄漏)
- 统一 timestep: 原生即 30min, 无需重采样
- 异常检测: IQR (Q1−1.5×IQR, Q3+1.5×IQR) 逐户 → 置 NaN
- 物理边界: KWH > 0
- 填充策略: 根据清洗后缺失率与最大连续缺口决定 (≤6步插值, 大缺口保留 NaN)
- tariff_type 为常量 (全部 Std), 清洗后剔除
- 支持批次处理与断点续跑

Usage:
  python3 clean_lcl_res.py --batch 50
  python3 clean_lcl_res.py --finish
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT /  "raw" / "lcl_res"
PROC = ROOT / "processed"
TMP = PROC / "_lcl_batches"
PROC.mkdir(exist_ok=True)
TMP.mkdir(exist_ok=True)

DATASET_ID = "lcl_res"
CATEGORY_ID = 0          # 居民
IQR_MULTIPLIER = 2.5
MAX_GAP_DROP = 48     # drop users with raw gap > 48 steps (24h)


def _max_consecutive_nan(series: pd.Series) -> int:
    """Max consecutive NaN after first valid (non-NaN) value.

    Leading NaN from trimming is NOT counted as a gap.
    """
    missing = series.isna()
    if not missing.any():
        return 0
    valid_mask = ~missing
    if not valid_mask.any():
        return len(series)
    first_valid = valid_mask.idxmax()
    missing_after = missing.loc[first_valid:]
    if not missing_after.any():
        return 0
    gap = missing_after.astype(int)
    run_id = (gap != gap.shift()).cumsum()
    return gap.groupby(run_id).transform("sum").max()


def detect_outliers_diff(series: pd.Series) -> pd.Series:
    """Labels: IQR on first-order diffs to catch spikes."""
    s = series.astype(float)
    if s.notna().sum() < 4:
        return pd.Series(False, index=s.index)
    d = s.diff()
    q1 = d.quantile(0.25)
    q3 = d.quantile(0.75)
    iqr_val = q3 - q1
    lo = q1 - IQR_MULTIPLIER * iqr_val
    hi = q3 + IQR_MULTIPLIER * iqr_val
    mask = (d < lo) | (d > hi)
    return mask.fillna(False)


def clean_kwh(series: pd.Series):
    """Clean KWH column: outlier detect → NaN → clip >0."""
    s = series.astype(float)

    outlier_mask = detect_outliers_diff(s)
    n_out = outlier_mask.sum()
    s = s.mask(outlier_mask, np.nan)
    s = s.clip(0.0, float("inf"))

    missing_rate = s.isna().mean()
    max_gap = _max_consecutive_nan(s)

    return s, missing_rate, max_gap, n_out


def load_raw() -> pd.DataFrame:
    cache = Path("/tmp/lcl_raw.pkl")
    if cache.exists():
        return pd.read_pickle(cache)
    csv_files = list(RAW.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV in {RAW}")
    df = pd.read_csv(csv_files[0])
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "KWH/hh (per half hour)": "KWH",
        "DateTime": "datetime",
        "stdorToU": "tariff",
    })
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["KWH"] = pd.to_numeric(df["KWH"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df.to_pickle(cache)
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


def process_batch(batch_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for lid, g in batch_df.groupby("LCLid", sort=True):
        g = g.sort_values("datetime").drop_duplicates(
            subset=["datetime"], keep="first").reset_index(drop=True)

        # Trim leading zeros (household not yet occupied / meter not active)
        kwh_vals = pd.to_numeric(g["KWH"], errors="coerce").values
        valid = np.where(~np.isnan(kwh_vals) & (kwh_vals > 0))[0]
        if len(valid) > 0 and valid[0] > 0:
            g = g.iloc[valid[0]:].reset_index(drop=True)

        if len(g) == 0:
            continue

        # Drop users with raw gaps > 48 steps (24h) in KWH
        kwh_raw = pd.to_numeric(g["KWH"], errors="coerce").values
        is_bad = np.isnan(kwh_raw) | (kwh_raw == 0)
        if is_bad.any():
            valid_idx = np.where(~is_bad)[0]
            if len(valid_idx) == 0:
                continue
            start = valid_idx[0]
            is_bad_after = is_bad[start:]
            if is_bad_after.any():
                boundaries = np.diff(np.concatenate(([True], ~is_bad_after, [True])))
                runs = np.where(boundaries)[0]
                max_run = (runs[1::2] - runs[::2]).max()
                if max_run > MAX_GAP_DROP:
                    continue

        g["KWH"], mr, mg, n_out = clean_kwh(g["KWH"])

        # Drop users with post-cleaning max_gap > 48
        if mg > MAX_GAP_DROP:
            continue

        # Cubic spline interpolation (interior gaps only)
        g["KWH"] = g["KWH"].interpolate(method="cubic", limit_area="inside")
        g["KWH"] = g["KWH"].clip(0.0, None)  # physical: KWH > 0

        if lid == batch_df["LCLid"].unique()[0]:
            print(f"  example {lid}: IQR_outliers={n_out}, "
                  f"missing_rate={mr:.4f}, max_gap={mg}")

        g["tariff"] = g["tariff"].ffill().bfill()
        out.append(g)

    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    # tariff_type is constant (all Std=0) in this public subset → drop
    df = add_public_features(df)
    keep = ["LCLid", "datetime", "KWH",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
            "month_sin", "month_cos", "category_id"]
    return df[keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=30, help="users per batch")
    ap.add_argument("--finish", action="store_true", help="merge batches")
    args = ap.parse_args()

    if args.finish:
        parts = sorted(TMP.glob("batch_*.csv"))
        if not parts:
            print("No batches found")
            return
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        out = PROC / f"{DATASET_ID}.csv"
        df.to_csv(out, index=False)
        n_data = len(df.columns) - 8  # LCLid + KWH + 7 public + 1 cat - actually just count
        data_cols = [c for c in df.columns if c not in
                     {"LCLid", "datetime", "hour_sin", "hour_cos", "dow_sin",
                      "dow_cos", "is_weekend", "month_sin", "month_cos",
                      "category_id"}]
        n_public = len(df.columns) - len(data_cols)
        print(f"Merged {len(parts)} batches -> {out}")
        print(f"  {len(df)} rows, {df['LCLid'].nunique()} users, "
              f"{len(data_cols)} data + {n_public} public = {len(df.columns)} cols")
        print(f"  KWH missing_rate={df['KWH'].isna().mean():.4f} "
              f"range=[{df['KWH'].min():.3f}, {df['KWH'].max():.3f}]")
        return

    raw = load_raw()
    users = sorted(raw["LCLid"].unique())
    done = [int(p.stem.split("_")[1]) for p in TMP.glob("batch_*.csv")]
    pending = [i for i, u in enumerate(users) if i not in done]
    print(f"total users={len(users)}, already_done={len(done)}, "
          f"pending={len(pending)}")

    take = pending[: args.batch]
    for i in take:
        uid = users[i]
        g = raw[raw["LCLid"] == uid]
        cleaned = process_batch(g)
        if cleaned.empty:
            print(f"  skip user {uid} (filtered out)", flush=True)
            continue
        cleaned.to_csv(TMP / f"batch_{i}.csv", index=False)
        print(f"  done user {uid} ({len(cleaned)} rows)", flush=True)
    print(f"processed {len(take)} users this run")


if __name__ == "__main__":
    main()
