#!/usr/bin/env python3
"""Clean script for tetouan_city (Power Consumption of Tetouan City, UCI).

- 统一 timestep 1800s (30min), resample=mean (原生 10min)
- 标签 (3 Zone 功率): IQR on first-order diffs (系数 1.96), 检测突变尖峰
- 特征 (5 气象): 滚动 IQR (30天窗口, 系数 1.96)
- 物理边界: 功率/气象 > 0
- 填充策略: 根据清洗后缺失率与最大连续缺口决定 (≤6步插值, 大缺口保留 NaN)
- 注意: category_id 清洗阶段写为 1 (配电网分区), 建模时 Zone1→2 工业, Zone2/3→0 居民
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT /  "raw" / "tetouan_city"
PROC = ROOT /  "processed"
PROC.mkdir(exist_ok=True)

DATASET_ID = "tetouan_city"
TARGET_TIMESTEP = 1800  # 30 min
CATEGORY_ID = 1          # 配电网分区 (建模时 Zone1→2 工业, Zone2/3→0 居民)
IQR_MULTIPLIER = 2.5
ROLLING_WINDOW = 336   # 7 days @ 30min
MAX_GAP_DROP = 72     # drop sequences with raw gap > 72 steps (24h)

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


def _max_consecutive_nan(series: pd.Series) -> int:
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
    iqr = q3 - q1
    lo = q1 - IQR_MULTIPLIER * iqr
    hi = q3 + IQR_MULTIPLIER * iqr
    mask = (d < lo) | (d > hi)
    return mask.fillna(False)


def detect_outliers_rolling(series: pd.Series) -> pd.Series:
    """Features: rolling IQR (window=7 days, center=True)."""
    s = series.astype(float)
    half = ROLLING_WINDOW // 2
    roll = s.rolling(window=ROLLING_WINDOW, center=True, min_periods=half)
    q1 = roll.quantile(0.25)
    q3 = roll.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - IQR_MULTIPLIER * iqr
    hi = q3 + IQR_MULTIPLIER * iqr
    mask = (s < lo) | (s > hi)
    return mask.fillna(False)


def clean_column(series: pd.Series, lo: float, hi: float,
                 detect_fn) -> tuple:
    s = series.astype(float)
    outlier_mask = detect_fn(s)
    n_out = outlier_mask.sum()
    s = s.mask(outlier_mask, np.nan)
    s = s.clip(lo, hi)

    missing_rate = s.isna().mean()
    max_gap = _max_consecutive_nan(s)

    return s, missing_rate, max_gap, n_out


def load_raw() -> pd.DataFrame:
    csv_files = list(RAW.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV in {RAW}")
    df = pd.read_csv(csv_files[0])
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


def clean() -> pd.DataFrame:
    df = load_raw()

    # Resample to 30min
    df = df.set_index("datetime")
    df = df.resample(f"{TARGET_TIMESTEP}s").mean(numeric_only=True)

    # ---- Drop columns with raw gaps > 48 steps in target zones ----
    for old_name, new_name in list(ZONE_TARGETS.items()):
        if old_name not in df.columns:
            continue
        vals = df[old_name].values.astype(np.float64)
        is_bad = np.isnan(vals) | (vals == 0)
        if is_bad.any():
            valid_idx = np.where(~is_bad)[0]
            if len(valid_idx) == 0:
                print(f"  WARNING: {old_name} all bad, dropping zone")
                del ZONE_TARGETS[old_name]
                continue
            start = valid_idx[0]
            is_bad_after = is_bad[start:]
            if is_bad_after.any():
                boundaries = np.diff(np.concatenate(([True], ~is_bad_after, [True])))
                runs = np.where(boundaries)[0]
                max_run = (runs[1::2] - runs[::2]).max()
                if max_run > MAX_GAP_DROP:
                    print(f"  WARNING: {old_name} raw max_gap={max_run} > "
                          f"{MAX_GAP_DROP}, dropping zone")
                    del ZONE_TARGETS[old_name]

    # ---- Trim leading zeros on all numeric columns ----
    for old_name in list(ZONE_TARGETS.keys()) + list(LOCAL_FEAT.keys()):
        if old_name not in df.columns:
            continue
        vals = df[old_name].values.astype(np.float64)
        valid = np.where(~np.isnan(vals) & (vals > 0))[0]
        if len(valid) > 0 and valid[0] > 0:
            df.iloc[:valid[0], df.columns.get_loc(old_name)] = np.nan

    # ---- Clean targets: diff IQR ----
    for old_name, new_name in list(ZONE_TARGETS.items()):
        if old_name in df.columns:
            df[new_name], mr, mg, n_out = clean_column(
                df[old_name], 0.0, float("inf"), detect_outliers_diff)
            print(f"  [target] {new_name}: diff_IQR_outliers={n_out}, "
                  f"missing_rate={mr:.4f}, max_gap={mg}")
            if mg > MAX_GAP_DROP:
                print(f"  WARNING: {new_name} post-cleaning max_gap={mg} > "
                      f"{MAX_GAP_DROP}, dropping zone")
                df = df.drop(columns=[old_name])
                del ZONE_TARGETS[old_name]

    # ---- Clean features: rolling IQR ----
    for old_name, new_name in list(LOCAL_FEAT.items()):
        if old_name in df.columns:
            df[new_name], mr, mg, n_out = clean_column(
                df[old_name], 0.0, float("inf"), detect_outliers_rolling)
            print(f"  [feature] {new_name}: rolling_IQR_outliers={n_out}, "
                  f"missing_rate={mr:.4f}, max_gap={mg}")
            if mg > MAX_GAP_DROP:
                print(f"  WARNING: {new_name} post-cleaning max_gap={mg} > "
                      f"{MAX_GAP_DROP}, dropping feature")
                df = df.drop(columns=[old_name])
                del LOCAL_FEAT[old_name]

    # ---- Cubic spline interpolation + re-clip bounds ----
    all_renamed = list(ZONE_TARGETS.values()) + list(LOCAL_FEAT.values())
    for c in all_renamed:
        if c in df.columns:
            df[c] = df[c].interpolate(method="cubic", limit_area="inside")
            df[c] = df[c].clip(0.0, None)  # power / weather > 0

    df = df.reset_index()
    df = add_public_features(df)

    keep = (["datetime"] + list(ZONE_TARGETS.values()) +
            list(LOCAL_FEAT.values()) +
            ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
             "month_sin", "month_cos", "category_id"])
    return df[keep]


def main() -> None:
    df = clean()
    out = PROC / f"{DATASET_ID}.csv"
    df.to_csv(out, index=False)
    data_cols = [c for c in df.columns if c not in
                 {"datetime", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                  "is_weekend", "month_sin", "month_cos", "category_id"}]
    n_public = len(df.columns) - len(data_cols)
    print(f"Wrote {out} ({len(df)} rows, "
          f"{len(data_cols)} data + {n_public} public = {len(df.columns)} cols)")
    for c in data_cols:
        print(f"  {c}: missing_rate={df[c].isna().mean():.4f} "
              f"range=[{df[c].min():.4f}, {df[c].max():.4f}]")


if __name__ == "__main__":
    main()
