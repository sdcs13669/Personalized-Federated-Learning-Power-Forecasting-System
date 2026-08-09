#!/usr/bin/env python3
"""Clean script for eld_ind (ElectricityLoadDiagrams 2011-2014, UCI).

Following 数据清洗规范 v1.0:
  - raw 只读,输出到 data/processed/eld_ind.csv
  - 统一 timestep 1800s (30min), resample=mean (从15min)
  - 缺失: 线性插值 (max_gap=6), 大缺口保留 NaN
  - 异常: 物理界 [0, 1e4] + MAD_Z + diff 突变
  - 向量化处理: 全 DataFrame 一次 resample/插值/异常, 规避逐列循环慢与 melt 内存爆炸
  - 特征工程: 公共时间特征 + 本地特征(无, §5.2.1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "eld_ind"
PROC = ROOT / "data" / "processed"
PROC.mkdir(exist_ok=True)

DATASET_ID = "eld_ind"
TARGET_TIMESTEP = 1800  # 30 min
CATEGORY_ID = 1  # 0=居民, 1=变压器, 2=工业
MAX_GAP = 6


def load_raw() -> pd.DataFrame:
    """Load raw eld_ind wide-format CSV (datetime + MT_001..MT_370)."""
    txt_files = list(RAW.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No TXT found in {RAW}")
    df = pd.read_csv(txt_files[0], sep=";", decimal=",", low_memory=False)
    df.columns = [c.strip().strip('"') for c in df.columns]
    ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    df = df.rename(columns={ts_col: "datetime"})
    return df


def fill_small_gaps_np(arr: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill missing values in each column if the NaN-run length <= max_gap.
    Column-wise linear interpolation; NaN runs longer than max_gap stay NaN.
    Vectorized across columns using pandas interpolate + run-length mask.
    """
    df = pd.DataFrame(arr)
    filled = df.interpolate(method="linear", limit_direction="both")
    miss = df.isna()
    out = filled.values.copy()
    # For each column, compute gap run lengths and mask back long runs
    for j in range(arr.shape[1]):
        s = df[j]
        m = miss[j]
        if not m.any():
            continue
        run_id = (m != m.shift()).cumsum()
        gap_len = m.groupby(run_id).transform("sum")
        fillable = m & (gap_len <= max_gap)
        out[~fillable.values, j] = np.nan
    return out


def clean() -> pd.DataFrame:
    df = load_raw()
    ts_col = "datetime"
    client_cols = [c for c in df.columns if c != ts_col]
    print(f"Wide-format: {len(df)} rows, {len(client_cols)} clients")

    # Resample all columns to 30min
    df = df.set_index(ts_col)
    df.index = pd.to_datetime(df.index)
    df = df[client_cols].resample(f"{TARGET_TIMESTEP}s").mean()
    print(f"After resample: {len(df)} rows")

    # --- Missing: fill small gaps via interpolation, keep large gaps NaN ---
    arr = df.values
    out = fill_small_gaps_np(arr, MAX_GAP)
    df = pd.DataFrame(out, index=df.index, columns=df.columns)

    # --- Outlier: physical clip + MAD_Z + diff spike ---
    X = df.values.astype(np.float64)
    X = np.clip(X, 0.0, 10000.0)
    # MAD robust Z per column
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0)
    mad[mad == 0] = 1.0
    z = 0.6745 * (X - med) / mad
    X[z > 10] = np.nan
    # diff spike
    d = np.abs(np.diff(X, axis=0, prepend=X[:1, :]))
    d_med = np.nanmedian(d, axis=0)
    d_mad = np.nanmedian(np.abs(d - d_med), axis=0)
    d_mad[d_mad == 0] = 1.0
    z_d = 0.6745 * (d - d_med) / d_mad
    X[z_d > 20] = np.nan
    # interpolate isolated outliers
    df = pd.DataFrame(X, index=df.index, columns=df.columns)
    df = df.interpolate(method="linear", limit_direction="both", limit=MAX_GAP)

    # Reset index
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: ts_col})

    # Add public features
    t = pd.to_datetime(df[ts_col])
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

    keep = (
        [ts_col] + client_cols +
        ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
         "month_sin", "month_cos", "category_id"]
    )
    return df[keep]


def main() -> None:
    df = clean()
    out = PROC / f"{DATASET_ID}.csv"
    df.to_csv(out, index=False)
    client_cols = [c for c in df.columns if c.startswith("MT_")]
    print(f"Wrote {out} ({len(df)} rows x {len(df.columns)} cols)")
    sub = df[client_cols]
    print(f"missing_rate={(sub.isna().mean().mean()):.4f} "
          f"range={sub.values.min():.2f}..{sub.values.max():.2f}")


if __name__ == "__main__":
    main()
