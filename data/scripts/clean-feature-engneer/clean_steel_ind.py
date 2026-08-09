#!/usr/bin/env python3
"""Clean script for steel_ind (Steel Industry Energy Consumption, UCI).

- 统一 timestep 1800s (30min), resample=mean (原生 15min)
- 标签 (Usage_kWh): IQR on first-order diffs (系数 1.96), 检测突变尖峰
- 特征 (无功功率/CO2/功率因数): 滚动 IQR (30天窗口, 系数 1.96)
- 物理边界: 功率/CO2 > 0; 功率因数 [0, 1] (从 % 转换为小数)
- 填充策略: 根据清洗后缺失率与最大连续缺口决定 (≤6步插值, 大缺口保留 NaN)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT /  "raw" / "steel_ind"
PROC = ROOT /  "processed"
PROC.mkdir(exist_ok=True)

DATASET_ID = "steel_ind"
TARGET_TIMESTEP = 1800  # 30 min
CATEGORY_ID = 2          # 钢铁工业
IQR_MULTIPLIER = 2.5
ROLLING_WINDOW = 336   # 7 days @ 30min
MAX_GAP_DROP = 48     # drop sequences with raw gap > 48 steps (24h)

TARGET_COL = "Usage_kWh"
FEATURE_COLS = [
    "lagging_reactive_power",
    "leading_reactive_power",
    "co2",
    "lagging_pf",
    "leading_pf",
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
    iqr_val = q3 - q1
    lo = q1 - IQR_MULTIPLIER * iqr_val
    hi = q3 + IQR_MULTIPLIER * iqr_val
    mask = (d < lo) | (d > hi)
    return mask.fillna(False)


def detect_outliers_rolling(series: pd.Series) -> pd.Series:
    """Features: rolling IQR (window=7 days, center=True)."""
    s = series.astype(float)
    half = ROLLING_WINDOW // 2
    roll = s.rolling(window=ROLLING_WINDOW, center=True, min_periods=half)
    q1 = roll.quantile(0.25)
    q3 = roll.quantile(0.75)
    iqr_val = q3 - q1
    lo = q1 - IQR_MULTIPLIER * iqr_val
    hi = q3 + IQR_MULTIPLIER * iqr_val
    mask = (s < lo) | (s > hi)
    return mask.fillna(False)


def clean_column(series: pd.Series, lo: float, hi: float,
                 detect_fn) -> tuple:
    """Clean one column: detect outliers, mark NaN, clip bounds.
    Returns (cleaned_series, missing_rate, max_gap, n_outliers).
    """
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
        raise FileNotFoundError(f"No CSV found in {RAW}")
    df = pd.read_csv(csv_files[0])
    df["datetime"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M",
                                     errors="coerce")
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

    # ---- Unit conversion (before resample) ----
    df["lagging_reactive_power"] = pd.to_numeric(
        df["Lagging_Current_Reactive.Power_kVarh"], errors="coerce")
    df["leading_reactive_power"] = pd.to_numeric(
        df["Leading_Current_Reactive_Power_kVarh"], errors="coerce")
    df["co2"] = pd.to_numeric(df["CO2(tCO2)"], errors="coerce")
    df["lagging_pf"] = pd.to_numeric(
        df["Lagging_Current_Power_Factor"], errors="coerce") / 100.0
    df["leading_pf"] = pd.to_numeric(
        df["Leading_Current_Power_Factor"], errors="coerce") / 100.0
    df["Usage_kWh"] = pd.to_numeric(df["Usage_kWh"], errors="coerce")

    # ---- Resample to 30min ----
    df = df.set_index("datetime")
    all_num = [TARGET_COL] + FEATURE_COLS
    df_num = df[all_num].resample(f"{TARGET_TIMESTEP}s").mean()
    df_cat = df[["Load_Type"]].resample(f"{TARGET_TIMESTEP}s").last().ffill()
    df = pd.concat([df_num, df_cat], axis=1)

    # ---- Drop sequences with raw gaps > 48 steps in target ----
    t_vals = df[TARGET_COL].values.astype(np.float64)
    is_bad = np.isnan(t_vals) | (t_vals == 0)
    if is_bad.any():
        valid_idx = np.where(~is_bad)[0]
        if len(valid_idx) == 0:
            print(f"ERROR: {TARGET_COL} all bad, aborting")
            return pd.DataFrame()
        start = valid_idx[0]
        is_bad_after = is_bad[start:]
        if is_bad_after.any():
            boundaries = np.diff(np.concatenate(([True], ~is_bad_after, [True])))
            runs = np.where(boundaries)[0]
            max_run = (runs[1::2] - runs[::2]).max()
            if max_run > MAX_GAP_DROP:
                print(f"ERROR: {TARGET_COL} has raw max_gap={max_run} > {MAX_GAP_DROP}, "
                      f"aborting")
                return pd.DataFrame()  # empty → caller skips

    # ---- Trim leading zeros on all numeric columns ----
    for col in [TARGET_COL] + FEATURE_COLS:
        vals = df[col].values.astype(np.float64)
        valid = np.where(~np.isnan(vals) & (vals > 0))[0]
        if len(valid) > 0 and valid[0] > 0:
            df.iloc[:valid[0], df.columns.get_loc(col)] = np.nan

    # ---- Clean target: diff IQR ----
    lo, hi = 0.0, float("inf")
    df[TARGET_COL], mr, mg, n_out = clean_column(df[TARGET_COL], lo, hi,
                                                  detect_outliers_diff)
    print(f"  [target] {TARGET_COL}: diff_IQR_outliers={n_out}, "
          f"missing_rate={mr:.4f}, max_gap={mg}")
    if mg > MAX_GAP_DROP:
        print(f"ERROR: {TARGET_COL} post-cleaning max_gap={mg} > {MAX_GAP_DROP}, "
              f"aborting")
        return pd.DataFrame()

    # ---- Clean features: rolling IQR ----
    bounds = {
        "lagging_reactive_power":    (0.0, float("inf")),
        "leading_reactive_power":    (0.0, float("inf")),
        "co2":                       (0.0, float("inf")),
        "lagging_pf":                (0.0, 1.0),
        "leading_pf":                (0.0, 1.0),
    }
    for col in list(FEATURE_COLS):
        lo, hi = bounds[col]
        df[col], mr, mg, n_out = clean_column(df[col], lo, hi,
                                               detect_outliers_rolling)
        print(f"  [feature] {col}: rolling_IQR_outliers={n_out}, "
              f"missing_rate={mr:.4f}, max_gap={mg}")
        if mg > MAX_GAP_DROP:
            print(f"  WARNING: {col} post-cleaning max_gap={mg} > "
                  f"{MAX_GAP_DROP}, dropping feature")
            df = df.drop(columns=[col])
            FEATURE_COLS.remove(col)

    # ---- Cubic spline interpolation + re-clip bounds ----
    interp_bounds = {
        "Usage_kWh":                  (0.0, float("inf")),
        "lagging_reactive_power":     (0.0, float("inf")),
        "leading_reactive_power":     (0.0, float("inf")),
        "co2":                        (0.0, float("inf")),
        "lagging_pf":                 (0.0, 1.0),
        "leading_pf":                 (0.0, 1.0),
    }
    for c, (lo, hi) in interp_bounds.items():
        if c in df.columns:
            df[c] = df[c].interpolate(method="cubic", limit_area="inside")
            df[c] = df[c].clip(lo, hi)

    # ---- Encode categorical ----
    load_map = {"Light_Load": 0, "Medium_Load": 1, "Maximum_Load": 2}
    df["load_type"] = df["Load_Type"].map(load_map).fillna(-1).astype(int)

    # ---- Public features ----
    df = df.reset_index()
    df = add_public_features(df)

    keep = ["datetime", "Usage_kWh",
            "lagging_reactive_power", "leading_reactive_power", "co2",
            "lagging_pf", "leading_pf", "load_type",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
            "month_sin", "month_cos", "category_id"]
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
