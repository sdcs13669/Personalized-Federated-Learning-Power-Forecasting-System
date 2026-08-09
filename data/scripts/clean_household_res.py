#!/usr/bin/env python3
"""Clean script for household_res (Individual household electric power consumption, UCI).

Following 数据清洗规范 v1.0:
  - 统一 timestep 1800s (30min) —— 原生 1min, resample=mean
  - 缺失: 线性插值 (max_gap=6), 大缺口保留 NaN (原始缺失以 '?' 表示)
  - 异常: 物理界 + MAD_Z + diff 突变
  - 特征工程: 公共时间特征 + 本地特征(无, 该数据集本地特征维度=0 per 清洗规范)
  - 注意: 单户数据集, 客户端=该户本身
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "household_res"
PROC = ROOT / "data" / "processed"
PROC.mkdir(exist_ok=True)

DATASET_ID = "household_res"
TARGET_TIMESTEP = 1800  # 30 min
CATEGORY_ID = 0  # 居民


def load_raw() -> pd.DataFrame:
    txt_files = list(RAW.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No txt in {RAW}")
    df = pd.read_csv(txt_files[0], sep=";", na_values=["?"])
    # Date/Time -> datetime
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
    df = df.drop(columns=["Date", "Time"])
    # 转数值
    for c in ["Global_active_power", "Global_reactive_power", "Voltage",
              "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
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


def clean_series(x: pd.Series, phys_lo: float, phys_hi: float) -> pd.Series:
    missing = x.isna()
    gap = missing.astype(int)
    group = (gap != gap.shift()).cumsum()
    gap_len = gap.groupby(group).transform("sum")
    fillable = missing & (gap_len <= 6)
    x_filled = x.interpolate(method="linear", limit_direction="both")
    x = x.where(~fillable, x_filled)

    x = x.clip(phys_lo, phys_hi)
    med = x.median()
    mad = (x - med).abs().median() or 1.0
    z = 0.6745 * (x - med) / mad
    x = x.mask(z.abs() > 10, np.nan)
    d = x.diff().abs()
    d_med = d.median()
    d_mad = (d - d_med).abs().median() or 1.0
    z_d = 0.6745 * (d - d_med) / d_mad
    x = x.mask(z_d > 20, np.nan)
    x = x.interpolate(method="linear", limit_direction="both", limit=6)
    return x


def clean() -> pd.DataFrame:
    df = load_raw()

    # 重采样到 30min
    df = df.set_index("datetime")
    df = df.resample(f"{TARGET_TIMESTEP}s").mean(numeric_only=True)

    # 对所有数值列做缺失+异常清洗(特征完整保留)
    for c in df.columns:
        if c in ["Global_active_power", "Global_reactive_power"]:
            df[c] = clean_series(df[c], 0.0, 50.0)
        else:
            df[c] = clean_series(df[c], -1e6, 1e6)

    df = df.reset_index()
    df = add_public_features(df)

    # 特征完整保留: 全部原始物理量列 + 公共特征, 不筛选
    return df


def main() -> None:
    df = clean()
    out = PROC / f"{DATASET_ID}.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")
    print(f"  missing_rate={df['Global_active_power'].isna().mean():.4f} "
          f"range={df['Global_active_power'].min():.4f}..{df['Global_active_power'].max():.4f}")


if __name__ == "__main__":
    main()
