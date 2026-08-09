#!/usr/bin/env python3
"""Clean script for household_res (Individual household electric power, UCI).

- 统一 timestep 1800s (30min), resample=mean (原生 1min)
- 标签 (Global_active_power): IQR on first-order diffs (系数 1.96), 检测突变尖峰
- 特征 (其他 6 数值列): 滚动 IQR (30天窗口, 系数 1.96)
- 物理边界: 全部 > 0
- 填充策略: 根据清洗后缺失率与最大连续缺口决定 (≤6步插值, 大缺口保留 NaN)
- 单户数据集, 不在客户端划分方案中, 作为参考保留
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT /  "raw" / "household_res"
PROC = ROOT / "processed"
PROC.mkdir(exist_ok=True)

DATASET_ID = "household_res"
TARGET_TIMESTEP = 1800  # 30 min
CATEGORY_ID = 0          # 居民
IQR_MULTIPLIER = 2.5
ROLLING_WINDOW = 336   # 7 days @ 30min
MAX_GAP_DROP = 336    # drop sequences with raw gap > 336 steps (7 days)
# NOTE: household_res has a ~5-day natural gap; threshold raised vs default 48

TARGET_COL = "Global_active_power"
FEATURE_COLS = [
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


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
    df = pd.read_csv(csv_files[0], na_values=["?"])
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
    df = df.drop(columns=["Date", "Time"])
    for c in [TARGET_COL] + FEATURE_COLS:
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


def clean() -> pd.DataFrame:
    df = load_raw()

    # Resample to 30min
    df = df.set_index("datetime")
    all_cols = [TARGET_COL] + FEATURE_COLS
    df = df[all_cols].resample(f"{TARGET_TIMESTEP}s").mean()

    # ---- Drop columns with raw gaps > 48 steps ----
    for c in all_cols:
        if c not in df.columns:
            continue
        vals = df[c].values.astype(np.float64)
        is_bad = np.isnan(vals) | (vals == 0)
        if is_bad.any():
            valid_idx = np.where(~is_bad)[0]
            if len(valid_idx) == 0:
                print(f"  WARNING: {c} all bad, dropping column")
                df = df.drop(columns=[c])
                if c in FEATURE_COLS:
                    FEATURE_COLS.remove(c)
                if c == TARGET_COL:
                    print(f"  ERROR: target column {c} dropped, aborting")
                    return pd.DataFrame()
                continue
            start = valid_idx[0]
            is_bad_after = is_bad[start:]
            if is_bad_after.any():
                boundaries = np.diff(np.concatenate(([True], ~is_bad_after, [True])))
                runs = np.where(boundaries)[0]
                max_run = (runs[1::2] - runs[::2]).max()
                if max_run > MAX_GAP_DROP:
                    print(f"  WARNING: {c} raw max_gap={max_run} > "
                          f"{MAX_GAP_DROP}, dropping column")
                    df = df.drop(columns=[c])
                    if c in FEATURE_COLS:
                        FEATURE_COLS.remove(c)
                    if c == TARGET_COL:
                        print(f"  ERROR: target column {c} dropped, aborting")
                        return pd.DataFrame()

    # ---- Trim leading zeros on all numeric columns ----
    for c in [TARGET_COL] + FEATURE_COLS:
        if c not in df.columns:
            continue
        vals = df[c].values.astype(np.float64)
        valid = np.where(~np.isnan(vals) & (vals > 0))[0]
        if len(valid) > 0 and valid[0] > 0:
            df.iloc[:valid[0], df.columns.get_loc(c)] = np.nan

    # ---- Clean target: diff IQR ----
    df[TARGET_COL], mr, mg, n_out = clean_column(
        df[TARGET_COL], 0.0, float("inf"), detect_outliers_diff)
    print(f"  [target] {TARGET_COL}: diff_IQR_outliers={n_out}, "
          f"missing_rate={mr:.4f}, max_gap={mg}")
    if mg > MAX_GAP_DROP:
        print(f"ERROR: {TARGET_COL} post-cleaning max_gap={mg} > "
              f"{MAX_GAP_DROP}, aborting")
        return pd.DataFrame()

    # ---- Clean features: rolling IQR ----
    for c in list(FEATURE_COLS):
        df[c], mr, mg, n_out = clean_column(
            df[c], 0.0, float("inf"), detect_outliers_rolling)
        print(f"  [feature] {c}: rolling_IQR_outliers={n_out}, "
              f"missing_rate={mr:.4f}, max_gap={mg}")
        if mg > MAX_GAP_DROP:
            print(f"  WARNING: {c} post-cleaning max_gap={mg} > "
                  f"{MAX_GAP_DROP}, dropping feature")
            df = df.drop(columns=[c])
            FEATURE_COLS.remove(c)

    # ---- Cubic spline interpolation + re-clip bounds (>0) ----
    for c in [TARGET_COL] + FEATURE_COLS:
        if c in df.columns:
            df[c] = df[c].interpolate(method="cubic", limit_area="inside")
            df[c] = df[c].clip(0.0, None)

    df = df.reset_index()
    df = add_public_features(df)
    return df


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
