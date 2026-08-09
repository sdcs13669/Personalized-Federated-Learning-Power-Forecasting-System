#!/usr/bin/env python3
"""Clean script for tetouan_city (Power Consumption of Tetouan City, UCI).

Following 数据清洗规范 v1.0:
  - 统一 timestep 1800s (30min), resample=mean
  - 缺失: 线性插值 (max_gap=6), 大缺口保留 NaN
  - 异常: 物理界 + MAD_Z + diff 突变
  - 特征工程: 公共时间特征 + 本地特征(5维)
  - 注意: 客户端按 Zone 划分(见清洗规范 §5.2.4),本脚本输出三个 Zone 的目标列。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "tetouan_city"
PROC = ROOT / "data" / "processed"
PROC.mkdir(exist_ok=True)

DATASET_ID = "tetouan_city"
TARGET_TIMESTEP = 1800  # 30 min
CATEGORY_ID = 1  # 0=居民, 1=变压器, 2=工业 (Zone是配电网分区,归为变压器/分区)

ZONE_TARGETS = {
    "Zone 1 Power Consumption": "load_zone1",
    "Zone 2  Power Consumption": "load_zone2",
    "Zone 3  Power Consumption": "load_zone3",
}

LOCAL_FEAT = {
    "Temperature": "temperature",
    "Humidity": "humidity",
    "Wind Speed": "wind_speed",
    "general diffuse flows": "general_diffuse_flow",
    "diffuse flows": "diffuse_flow",
}


def load_raw() -> pd.DataFrame:
    csv_files = list(RAW.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV in {RAW}")
    df = pd.read_csv(csv_files[0])
    # DateTime like '1/1/2017 0:00'
    df["datetime"] = pd.to_datetime(df["DateTime"], errors="coerce")
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
    """缺失 + 异常处理通用流程."""
    missing = x.isna()
    gap = missing.astype(int)
    group = (gap != gap.shift()).cumsum()
    gap_len = gap.groupby(group).transform("sum")
    fillable = missing & (gap_len <= 6)
    x_filled = x.interpolate(method="linear", limit_direction="both")
    x = x.where(~fillable, x_filled)

    # physical bounds
    x = x.clip(phys_lo, phys_hi)
    # MAD robust z
    med = x.median()
    mad = (x - med).abs().median() or 1.0
    z = 0.6745 * (x - med) / mad
    x = x.mask(z.abs() > 10, np.nan)
    # diff spike
    d = x.diff().abs()
    d_med = d.median()
    d_mad = (d - d_med).abs().median() or 1.0
    z_d = 0.6745 * (d - d_med) / d_mad
    x = x.mask(z_d > 20, np.nan)
    # fill small isolated gaps
    x = x.interpolate(method="linear", limit_direction="both", limit=6)
    return x


def clean() -> pd.DataFrame:
    df = load_raw()

    # timestep to 30min
    df = df.set_index("datetime")
    df = df.resample(f"{TARGET_TIMESTEP}s").mean(numeric_only=True)

    # clean each zone power (target), physical bounds in W
    # 特征完整保留: 保留原始列名, 同时生成标准命名副本
    for col, new in ZONE_TARGETS.items():
        df[new] = clean_series(df[col], 0.0, 1e7)

    # clean local features (保留原始列 + 标准命名副本)
    for col, new in LOCAL_FEAT.items():
        if col in df.columns:
            df[new] = clean_series(df[col], -1e6, 1e6)

    df = df.reset_index()
    df = add_public_features(df)

    # 特征完整保留: 所有原始列(气象+三Zone功率) + 标准命名副本 + 公共特征
    return df


def main() -> None:
    df = clean()
    out = PROC / f"{DATASET_ID}.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")
    for c in ZONE_TARGETS.values():
        print(f"  {c}: missing_rate={df[c].isna().mean():.4f} "
              f"range={df[c].min():.1f}..{df[c].max():.1f}")


if __name__ == "__main__":
    main()
