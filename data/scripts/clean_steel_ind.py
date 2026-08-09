#!/usr/bin/env python3
"""Clean script for steel_ind (Steel Industry Energy Consumption, UCI).

Following 数据清洗规范 v1.0:
  - raw 只读,输出到 data/processed/steel_ind.csv
  - 统一 timestep 1800s (30min), resample=mean
  - 缺失: 线性插值 (max_gap=6), 大缺口保留 NaN
  - 异常: 物理界 [0, 1e6] + MAD_Z + diff 突变
  - 特征工程: 公共时间特征 + 本地特征(6维)
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "steel_ind"
PROC = ROOT / "data" / "processed"
PROC.mkdir(exist_ok=True)

DATASET_ID = "steel_ind"
TARGET_TIMESTEP = 1800  # 30 min
CATEGORY_ID = 2  # 0=居民, 1=变压器, 2=工业


def load_raw() -> pd.DataFrame:
    """Load raw steel_ind CSV, unify timestamp."""
    # locate csv (either in zip-extracted dir or direct file)
    csv_files = list(RAW.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {RAW}")
    df = pd.read_csv(csv_files[0])
    # date like '01/01/2018 00:15' -> datetime
    df["datetime"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M", errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df


def add_public_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 8 public encoder features from timestamp (清洗规范 §5.1)."""
    t = df["datetime"]
    hour = t.dt.hour + t.dt.minute / 60.0
    dow = t.dt.dayofweek  # Monday=0
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


def local_features(df: pd.DataFrame) -> pd.DataFrame:
    """Steel local features (§5.2.3): 6-dim.
    特征完整保留: 仅对列做规范化命名, 不筛选任何原始列。
    """
    df = df.rename(columns={
        "Lagging_Current_Reactive.Power_kVarh": "lagging_reactive_power",
        "Leading_Current_Reactive_Power_kVarh": "leading_reactive_power",
        "CO2(tCO2)": "co2",
        "Lagging_Current_Power_Factor": "lagging_pf",
        "Leading_Current_Power_Factor": "leading_pf",
    })
    # Load_Type 保留原始类别,同时生成数值编码
    load_map = {"Light_Load": 0, "Medium_Load": 1, "Maximum_Load": 2}
    df["load_type"] = df["Load_Type"].map(load_map).fillna(-1).astype(int)
    return df


def clean() -> pd.DataFrame:
    df = load_raw()

    # --- timestep unify to 30min ---
    df = df.set_index("datetime")
    # resample mean to 30min (from native 15min)
    df_num = df.select_dtypes(include=[np.number]).resample(f"{TARGET_TIMESTEP}s").mean()
    # categorical columns: take last value in each 30min window (ffill)
    df_cat = df.select_dtypes(exclude=[np.number]).resample(f"{TARGET_TIMESTEP}s").last().ffill()
    df = pd.concat([df_num, df_cat], axis=1)

    # --- missing: linear interp with max_gap ---
    target = df["Usage_kWh"]
    missing = target.isna()
    # gap length per NaN run
    gap = missing.astype(int)
    group = (gap != gap.shift()).cumsum()
    gap_len = gap.groupby(group).transform("sum")
    fillable = missing & (gap_len <= 6)
    # linear interpolation only for fillable positions
    target_filled = target.interpolate(method="linear", limit_direction="both")
    target = target.where(~fillable, target_filled)
    df["Usage_kWh"] = target

    # --- outlier: physical bounds + MAD_Z + diff ---
    x = target
    # physical
    phys_lo, phys_hi = 0.0, 1e6
    x = x.clip(phys_lo, phys_hi)
    # MAD robust z
    med = x.median()
    mad = (x - med).abs().median() or 1.0
    z = 0.6745 * (x - med) / mad
    x = x.mask(z.abs() > 10, np.nan)  # beyond threshold -> NaN
    # diff spike detection
    d = x.diff().abs()
    d_med = d.median()
    d_mad = (d - d_med).abs().median() or 1.0
    z_d = 0.6745 * (d - d_med) / d_mad
    x = x.mask(z_d > 20, np.nan)
    # fill isolated outliers by interpolation (small gaps)
    x = x.interpolate(method="linear", limit_direction="both", limit=6)
    df["Usage_kWh"] = x

    # --- public + local features ---
    df = df.reset_index()
    df = add_public_features(df)
    df = local_features(df)

    # 特征完整保留: 所有原始列 + 工程特征, 不筛选
    # 原始列: date(已转datetime), Usage_kWh, Lagging/Leading_*, CO2, 功率因数, NSM, WeekStatus, Day_of_week, Load_Type
    # 工程特征: 8公共特征 + load_type(数值编码)
    return df


def main() -> None:
    df = clean()
    out = PROC / f"{DATASET_ID}.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")
    # quick quality metrics
    print(f"rows={len(df)} missing_rate={(df['Usage_kWh'].isna().mean()):.4f} "
          f"range={df['Usage_kWh'].min():.2f}..{df['Usage_kWh'].max():.2f}")


if __name__ == "__main__":
    main()
