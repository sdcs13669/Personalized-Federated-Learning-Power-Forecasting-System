#!/usr/bin/env python3
"""Clean script for lcl_res (Low Carbon London smart meter, London Datastore).

Following 数据清洗规范 v1.0:
  - 按客户端(每户 LCLid)独立处理缺失/异常, 不跨客户端借信息 (§8)
  - 缺失: 线性插值 (max_gap=6), 大缺口保留 NaN
  - 异常: 物理界 [0, 100] kWh/hh + MAD_Z + diff 突变
  - 特征工程: 公共时间特征 + 本地特征 tariff_type(1维)

Usage:
  python3 clean_lcl_res.py --batch 50     # 每批处理50户, 可多次运行续跑
  python3 clean_lcl_res.py --finish       # 合并所有批次并输出最终csv
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "lcl_res"
PROC = ROOT / "data" / "processed"
TMP = PROC / "_lcl_batches"
PROC.mkdir(exist_ok=True)
TMP.mkdir(exist_ok=True)

DATASET_ID = "lcl_res"
CATEGORY_ID = 0  # 居民


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


def clean_series(x: pd.Series, phys_lo: float, phys_hi: float) -> pd.Series:
    x = x.copy().astype(float)
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


def process_batch(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Clean one batch of users (already subset by user)."""
    out = []
    for lid, g in batch_df.groupby("LCLid", sort=True):
        g = g.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="first")
        g = g.reset_index(drop=True)
        g["KWH"] = clean_series(g["KWH"], 0.0, 100.0)
        g["tariff"] = g["tariff"].ffill().bfill()
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    df["tariff_type"] = df["tariff"].map({"Std": 0, "ToU": 1}).fillna(0).astype(int)
    df = add_public_features(df)
    keep = ["LCLid", "datetime", "KWH", "tariff_type",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
            "month_sin", "month_cos", "category_id"]
    return df[keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=30, help="users per batch")
    ap.add_argument("--finish", action="store_true", help="merge batches into final csv")
    args = ap.parse_args()

    if args.finish:
        parts = sorted(TMP.glob("batch_*.csv"))
        if not parts:
            print("No batches found")
            return
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        out = PROC / f"{DATASET_ID}.csv"
        df.to_csv(out, index=False)
        print(f"Merged {len(parts)} batches -> {out} ({len(df)} rows, "
              f"{df['LCLid'].nunique()} users)")
        print(f"  missing_rate={df['KWH'].isna().mean():.4f} "
              f"range={df['KWH'].min():.3f}..{df['KWH'].max():.3f}")
        return

    raw = load_raw()
    users = sorted(raw["LCLid"].unique())
    done = [int(p.stem.split("_")[1]) for p in TMP.glob("batch_*.csv")]
    pending = [i for i, u in enumerate(users) if i not in done]
    print(f"total users={len(users)}, already_done={len(done)}, pending={len(pending)}")

    take = pending[: args.batch]
    for i in take:
        uid = users[i]
        g = raw[raw["LCLid"] == uid]
        cleaned = process_batch(g)
        cleaned.to_csv(TMP / f"batch_{i}.csv", index=False)
        print(f"  done user {uid} ({len(cleaned)} rows)", flush=True)
    print(f"processed {len(take)} users this run")


if __name__ == "__main__":
    main()
