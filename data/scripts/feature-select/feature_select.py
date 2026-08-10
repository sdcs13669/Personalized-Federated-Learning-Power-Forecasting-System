#!/usr/bin/env python3
"""Feature selection using XGBoost + SHAP for power load forecasting datasets.

Reads processed CSV (read-only), runs statistical filters + XGBoost + SHAP,
and outputs a YAML report to data/feature_selection/.

Usage:
  python feature_select.py --input data/processed/steel_ind.csv --target Usage_kWh
  python feature_select.py --input data/processed/tetouan_city.csv --target load_zone1
  python feature_select.py --input data/processed/tetouan_city.csv --target load_zone2
  python feature_select.py --input data/processed/tetouan_city.csv --target load_zone3
"""
from __future__ import annotations

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

# Columns excluded from feature candidates (not target, not datetime)
SKIP_COLS = {"datetime", "DateTime", "date", "Date", "Time", "timestamp"}

# Public features that form the fixed encoder input (§5.2.1).
# These should NOT be dropped regardless of SHAP score or VIF,
# because they are shared across all datasets in the FL architecture.
PUBLIC_FEATURES = {
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "month_sin", "month_cos", "category_id",
}


def _detect_dataset_id(csv_path: str) -> str:
    return Path(csv_path).stem


def _detect_other_targets(df: pd.DataFrame, target: str) -> list[str]:
    """Find columns that share the target's base name (e.g. load_zone2,3 for load_zone1)."""
    base = re.sub(r'\d+$', '', target)
    if len(base) < 3:
        return []
    return [c for c in df.columns if c != target and c.startswith(base)]


def _get_feature_cols(df: pd.DataFrame, target: str) -> list[str]:
    """Return candidate feature columns: all numeric cols except target, datetime, other targets."""
    other_targets = set(_detect_other_targets(df, target))
    skip = SKIP_COLS | other_targets | {target}
    return [c for c in df.columns if c not in skip]


# ---------------------------------------------------------------------------
# Statistical filters
# ---------------------------------------------------------------------------

def missing_rate_report(df: pd.DataFrame, features: list[str]) -> dict:
    """Per-feature missing rate."""
    rates = {}
    for c in features:
        mr = df[c].isna().mean()
        rates[c] = round(float(mr), 4)
    return rates


def constant_report(df: pd.DataFrame, features: list[str]) -> list[str]:
    """Return features with near-zero variance (single unique value)."""
    const = []
    for c in features:
        vals = df[c].dropna()
        if vals.nunique() <= 1:
            const.append(c)
    return const


def correlation_report(df: pd.DataFrame, features: list[str],
                       threshold: float = 0.9) -> list[dict]:
    """Return highly correlated feature pairs (|r| > threshold)."""
    corr = df[features].corr()
    pairs = []
    seen = set()
    for i, c1 in enumerate(features):
        for c2 in features[i + 1:]:
            r = corr.loc[c1, c2]
            if pd.notna(r) and abs(r) > threshold and (c2, c1) not in seen:
                pairs.append({"pair": (c1, c2), "correlation": round(float(r), 4)})
                seen.add((c1, c2))
    return pairs


def vif_report(df: pd.DataFrame, features: list[str]) -> dict:
    """Compute VIF for each feature after dropping NaN rows."""
    sub = df[features].dropna()
    if len(sub) < 10:
        return {"error": "too few complete rows for VIF"}
    # Drop constant columns (VIF undefined)
    sub = sub.loc[:, sub.nunique() > 1]
    if sub.shape[1] < 2:
        return {"message": "fewer than 2 non-constant features, VIF skipped"}
    X = sub.values.astype(np.float64)
    # Add constant
    X_c = np.column_stack([np.ones(X.shape[0]), X])
    vif_vals = {}
    for j, col in enumerate(sub.columns):
        try:
            vif_vals[col] = round(float(variance_inflation_factor(X_c, j + 1)), 2)
        except Exception:
            vif_vals[col] = None
    return vif_vals


# ---------------------------------------------------------------------------
# XGBoost + SHAP
# ---------------------------------------------------------------------------

def _chrono_split(df: pd.DataFrame, features: list[str], target: str):
    """Chronological 80/20 split (no shuffle, no validation set)."""
    n = len(df)

    X_all = df[features].copy()
    y_all = df[target].copy()

    # Drop rows where target is NaN
    valid = y_all.notna()
    X_all = X_all[valid]
    y_all = y_all[valid]
    # Re-align chronological indices
    n_valid = len(y_all)
    train_end = int(n_valid * 0.80)

    X_train = X_all.iloc[:train_end]
    y_train = y_all.iloc[:train_end]
    X_test = X_all.iloc[train_end:]
    y_test = y_all.iloc[train_end:]

    return X_train, X_test, y_train, y_test, features


def _train_xgboost(X_train, y_train):
    """Train XGBoost regressor (fixed estimators, no early stopping)."""
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def _shap_analysis(model, X_test, dataset_id, target):
    """Compute SHAP values, generate plots. Returns SHAP importance dict."""
    out_dir = FIG_DIR / dataset_id
    out_dir.mkdir(exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test, check_additivity=False)

    # -- Bar plot (mean |SHAP|) --
    fig, ax = plt.subplots(figsize=(10, max(5, len(X_test.columns) * 0.3)))
    shap.plots.bar(shap_values, show=False)
    fig.tight_layout()
    bar_path = out_dir / f"{dataset_id}_{target}_shap_bar.png"
    fig.savefig(bar_path)
    plt.close(fig)
    print(f"  -> {bar_path}")

    # -- Beeswarm summary --
    fig, ax = plt.subplots(figsize=(10, max(5, len(X_test.columns) * 0.3)))
    shap.plots.beeswarm(shap_values, show=False)
    fig.tight_layout()
    swarm_path = out_dir / f"{dataset_id}_{target}_shap_summary.png"
    fig.savefig(swarm_path)
    plt.close(fig)
    print(f"  → {swarm_path}")

    # -- Mean |SHAP| ranking --
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    ranking = sorted(
        zip(X_test.columns, mean_shap),
        key=lambda x: x[1], reverse=True
    )
    return {col: round(float(val), 6) for col, val in ranking}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_report(dataset_id: str, target: str, features: list[str],
                 missing: dict, constants: list[str],
                 high_corr: list[dict], vif: dict,
                 metrics: dict, shap_ranking: dict,
                 xgb_params: dict):
    """Write YAML feature selection report."""
    lines = []
    lines.append(f"# {dataset_id}/{target} XGBoost+SHAP 特征筛选报告")
    lines.append(f"# 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("meta:")
    lines.append(f"  dataset_id: {dataset_id}")
    lines.append(f"  target: {target}")
    lines.append(f"  script: scripts/feature-select/feature_select.py")
    lines.append(f"  n_features: {len(features)}")
    lines.append(f"  n_samples: {missing.get('_n_total', '?')}")
    lines.append("")
    lines.append("# ---- 1. 统计筛选 ----")
    lines.append("statistical:")
    lines.append(f"  constants: {constants}")
    lines.append("  high_correlation_pairs:")
    if high_corr:
        for p in high_corr:
            lines.append(f"    - pair: [{p['pair'][0]}, {p['pair'][1]}]")
            lines.append(f"      correlation: {p['correlation']}")
    else:
        lines.append("    []")
    lines.append("  vif:")
    if isinstance(vif, dict) and "error" not in vif and "message" not in vif:
        for col, val in vif.items():
            flag = "  # HIGH" if (val is not None and val > 10) else ""
            lines.append(f"    {col}: {val}{flag}")
    else:
        lines.append(f"    note: {vif}")
    lines.append("  missing_rate:")
    for col, rate in sorted(missing.items(), key=lambda x: x[1], reverse=True):
        if col.startswith("_"):
            continue
        flag = "  # HIGH" if rate > 0.5 else ""
        lines.append(f"    {col}: {rate}{flag}")
    lines.append("")
    lines.append("# ---- 2. XGBoost 性能 ----")
    lines.append("xgb_performance:")
    for k, v in metrics.items():
        lines.append(f"  {k}: {v}")
    lines.append(f"  params: {xgb_params}")
    lines.append("")
    lines.append("# ---- 3. SHAP 特征重要性 ----")
    lines.append("shap_importance:")
    for col, val in shap_ranking.items():
        lines.append(f"  {col}: {val}")
    lines.append("")
    lines.append("# ---- 4. 筛选建议 ----")
    lines.append("recommendations:")

    # Generate automated recommendations.
    # Public features are excluded from auto-drop because they form the
    # fixed encoder input shared across all clients in the FL architecture.
    drop = []
    drop_cols_set = set()
    retain = []

    def _add_drop(col, reason):
        if col not in drop_cols_set and col not in PUBLIC_FEATURES:
            drop.append(f"{col}: {reason}")
            drop_cols_set.add(col)

    # Constants -> drop
    for c in constants:
        _add_drop(c, "constant/near-zero variance, no information")
    # High VIF -> flag
    if isinstance(vif, dict) and "error" not in vif and "message" not in vif:
        for col, val in vif.items():
            if val is not None and val > 10:
                reason = "VIF>10"
                for p in high_corr:
                    if col in p["pair"]:
                        other = p["pair"][0] if p["pair"][1] == col else p["pair"][1]
                        reason += f", highly correlated with {other} (r={p['correlation']})"
                        break
                _add_drop(col, reason)
    # Low SHAP -> flag
    total_shap = sum(shap_ranking.values())
    if total_shap > 0:
        for col, val in shap_ranking.items():
            if val / total_shap < 0.01:
                _add_drop(col, f"SHAP contribution < 1% ({val:.6f})")

    # Remaining -> retain
    for col in features:
        if col not in drop_cols_set:
            if col in shap_ranking:
                retain.append(f"{col}: SHAP={shap_ranking[col]:.6f}")
            else:
                retain.append(f"{col}: (not in model)")
    # Add public features note
    public_in_features = [c for c in features if c in PUBLIC_FEATURES]
    if public_in_features:
        retain.append(f"# public features (encoder input, always retained): "
                      f"{', '.join(public_in_features)}")

    lines.append("  drop:")
    if drop:
        for d in drop:
            lines.append(f"    - {d}")
    else:
        lines.append("    []")
    lines.append("  retain:")
    for r in retain:
        lines.append(f"    - {r}")

    out_path = REPORT_DIR / f"{dataset_id}_{target}_shap.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="XGBoost + SHAP feature selection for power load datasets")
    ap.add_argument("--input", required=True,
                    help="Path to processed CSV file")
    ap.add_argument("--target", required=True,
                    help="Target column name")
    ap.add_argument("--exclude", default=None,
                    help="Comma-separated columns to exclude from features")
    ap.add_argument("--no-plots", action="store_true",
                    help="Skip SHAP plot generation")
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    dataset_id = _detect_dataset_id(args.input)
    target = args.target

    if target not in df.columns:
        print(f"ERROR: target '{target}' not found in columns")
        return

    # Feature detection
    features = _get_feature_cols(df, target)
    if args.exclude:
        extra = [c.strip() for c in args.exclude.split(",")]
        features = [c for c in features if c not in extra]
        print(f"Excluded: {extra}")

    print(f"Dataset: {dataset_id} | Target: {target}")
    print(f"Features ({len(features)}): {', '.join(features)}")
    print(f"Samples: {len(df)}")

    # -- 1. Statistical analysis --
    print("\n-- Statistical filters --")
    missing = missing_rate_report(df, features)
    missing["_n_total"] = len(df)
    for c, r in sorted(missing.items(), key=lambda x: x[1], reverse=True):
        if not c.startswith("_"):
            print(f"  missing_rate: {c}={r:.4f}")

    constants = constant_report(df, features)
    if constants:
        print(f"  CONSTANT: {constants}")

    high_corr = correlation_report(df, features)
    if high_corr:
        print(f"  HIGH CORRELATION ({len(high_corr)} pairs):")
        for p in high_corr:
            print(f"    {p['pair'][0]} <-> {p['pair'][1]}: r={p['correlation']}")

    vif = vif_report(df, features)
    if isinstance(vif, dict) and "error" not in vif and "message" not in vif:
        high_vif = {c: v for c, v in vif.items() if v is not None and v > 10}
        if high_vif:
            print(f"  HIGH VIF (>10): {high_vif}")
    else:
        print(f"  VIF: {vif}")

    # -- 2. XGBoost training --
    print("\n-- XGBoost training --")
    X_train, X_test, y_train, y_test, feature_list = _chrono_split(
        df, features, target)

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    model = _train_xgboost(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    metrics = {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
    }
    print(f"  MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    xgb_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    # -- 3. SHAP analysis --
    print("\n-- SHAP analysis --")
    shap_ranking = {}
    if not args.no_plots:
        shap_ranking = _shap_analysis(model, X_test, dataset_id, target)
    else:
        # Still compute ranking without plots
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test, check_additivity=False)
        mean_shap = np.abs(shap_values.values).mean(axis=0)
        shap_ranking = {
            col: round(float(val), 6)
            for col, val in sorted(
                zip(X_test.columns, mean_shap),
                key=lambda x: x[1], reverse=True
            )
        }

    print("  SHAP importance ranking:")
    for col, val in shap_ranking.items():
        print(f"    {col}: {val:.6f}")

    # -- 4. Write report --
    print("\n-- Report --")
    write_report(dataset_id, target, feature_list,
                 missing, constants, high_corr, vif,
                 metrics, shap_ranking, xgb_params)

    print("\nDone.")


if __name__ == "__main__":
    main()
