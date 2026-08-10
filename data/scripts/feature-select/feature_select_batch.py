#!/usr/bin/env python3
"""Batch feature selection: run XGBoost+SHAP on all target sequences in a
wide-format dataset, then average SHAP importance across sequences.

Usage:
  python feature_select_batch.py --input data/processed/eld_ind.csv
  python feature_select_batch.py --input data/processed/lcl_res.csv
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import xgboost as xgb
import shap

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "data" / "figures"
REPORT_DIR = ROOT / "data" / "feature_selection"
REPORT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
})

SKIP_COLS = {"datetime", "DateTime", "date", "Date", "Time", "timestamp"}

PUBLIC_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "month_sin", "month_cos", "category_id",
]

# XGBoost params (lightweight for batch processing)
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}


def _detect_dataset_id(csv_path: str) -> str:
    return Path(csv_path).stem


def _detect_target_cols(df: pd.DataFrame) -> list[str]:
    """Find all target columns (not datetime, not public features)."""
    skip = SKIP_COLS | set(PUBLIC_FEATURES)
    return [c for c in df.columns if c not in skip]


def _train_and_shap(X, y) -> tuple[dict, float]:
    """Train XGBoost and return mean |SHAP| dict + test R2. Fast path, no plots."""
    # Drop NaN targets
    valid = y.notna()
    X_valid = X[valid].values
    y_valid = y[valid].values
    if len(y_valid) < 100:
        return {}, float("nan")

    n = len(y_valid)
    train_end = int(n * 0.80)

    X_train, y_train = X_valid[:train_end], y_valid[:train_end]
    X_test, y_test = X_valid[train_end:], y_valid[train_end:]

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)

    y_pred = model.predict(X_test)
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X_test, check_additivity=False)
    mean_shap = np.abs(shap_vals.values).mean(axis=0)
    return {col: float(v) for col, v in zip(X.columns, mean_shap)}, float(r2)


def main():
    ap = argparse.ArgumentParser(
        description="Batch XGBoost+SHAP across all target sequences")
    ap.add_argument("--input", required=True, help="Path to processed CSV")
    ap.add_argument("--max-targets", type=int, default=0,
                    help="Limit number of targets (0=all)")
    ap.add_argument("--max-nan-rate", type=float, default=0.5,
                    help="Skip targets with NaN rate above this threshold")
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    dataset_id = _detect_dataset_id(args.input)
    target_cols = _detect_target_cols(df)
    n_total = len(target_cols)

    # Ensure public features exist
    features = [c for c in PUBLIC_FEATURES if c in df.columns]
    print(f"Dataset: {dataset_id}")
    print(f"Target columns: {n_total}")
    print(f"Features ({len(features)}): {features}")
    print(f"Rows: {len(df)}")

    X = df[features].copy()

    # Pre-filter targets by NaN rate
    valid_targets = []
    for t in target_cols:
        mr = df[t].isna().mean()
        if mr <= args.max_nan_rate:
            valid_targets.append(t)

    if args.max_targets > 0:
        valid_targets = valid_targets[:args.max_targets]

    n_valid = len(valid_targets)
    print(f"Targets with <= {args.max_nan_rate:.0%} NaN: {n_valid}")
    print(f"Processing...")

    # Accumulate SHAP values per feature
    shap_accum = {f: [] for f in features}
    r2_list = []
    n_done = 0
    n_failed = 0

    for i, target in enumerate(valid_targets):
        shap_dict, r2 = _train_and_shap(X, df[target])
        if shap_dict:
            for f in features:
                shap_accum[f].append(shap_dict.get(f, 0.0))
            r2_list.append(r2)
            n_done += 1
        else:
            n_failed += 1

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_valid} done, {n_failed} skipped")

    print(f"Completed: {n_done} succeeded, {n_failed} skipped")

    if n_done == 0:
        print("ERROR: no valid targets processed")
        return

    # ---- Aggregate results ----
    mean_shap = {f: float(np.mean(shap_accum[f])) for f in features}
    std_shap = {f: float(np.std(shap_accum[f])) for f in features}
    median_r2 = float(np.median(r2_list))
    mean_r2 = float(np.mean(r2_list))

    # Ranking by mean SHAP
    ranking = sorted(mean_shap.items(), key=lambda x: x[1], reverse=True)

    print(f"\nAggregate SHAP importance (mean across {n_done} targets):")
    for col, val in ranking:
        print(f"  {col}: {val:.6f} +/- {std_shap[col]:.6f}")
    print(f"\nR2: mean={mean_r2:.4f}, median={median_r2:.4f}")

    # ---- Aggregate bar plot ----
    out_dir = FIG_DIR / dataset_id
    out_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#1f77b4" if "hour" in c or "dow" in c or "month" in c
              else "#ff7f0e" for c in [r[0] for r in ranking]]
    bars = ax.barh(
        [r[0] for r in reversed(ranking)],
        [r[1] for r in reversed(ranking)],
        xerr=[std_shap[r[0]] for r in reversed(ranking)],
        color=[colors[len(ranking) - 1 - i] for i in range(len(ranking))],
        capsize=2,
    )
    ax.set_xlabel("Mean |SHAP| value")
    ax.set_title(f"{dataset_id} — aggregate SHAP across {n_done} targets "
                 f"(mean R2={mean_r2:.3f})")
    fig.tight_layout()
    plot_path = out_dir / f"{dataset_id}_aggregate_shap_bar.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"  -> {plot_path}")

    # ---- Aggregate report ----
    lines = []
    lines.append(f"# {dataset_id} 聚合 SHAP 特征重要性报告")
    lines.append(f"# 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"# 对 {n_done} 个目标序列逐列 XGBoost+SHAP 后取均值")
    lines.append("")
    lines.append("meta:")
    lines.append(f"  dataset_id: {dataset_id}")
    lines.append(f"  script: scripts/feature-select/feature_select_batch.py")
    lines.append(f"  n_targets_total: {n_total}")
    lines.append(f"  n_targets_processed: {n_done}")
    lines.append(f"  n_targets_failed: {n_failed}")
    lines.append(f"  max_nan_rate_filter: {args.max_nan_rate}")
    lines.append(f"  n_features: {len(features)}")
    lines.append(f"  n_rows: {len(df)}")
    lines.append("")
    lines.append("aggregate_performance:")
    lines.append(f"  r2_mean: {mean_r2:.4f}")
    lines.append(f"  r2_median: {median_r2:.4f}")
    lines.append("")
    lines.append("aggregate_shap_importance:")
    for col, val in ranking:
        lines.append(f"  {col}:")
        lines.append(f"    mean: {val:.6f}")
        lines.append(f"    std: {std_shap[col]:.6f}")
    lines.append("")
    lines.append("recommendations:")
    lines.append(f"  note: >")
    lines.append(f"    All {len(features)} public features are the fixed encoder input. ")
    lines.append(f"    No local features exist in this wide-format dataset. ")
    lines.append(f"    The SHAP ranking above reflects which time features are most ")
    lines.append(f"    informative on average across the {n_done} target sequences.")

    report_path = REPORT_DIR / f"{dataset_id}_aggregate_shap.yaml"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
