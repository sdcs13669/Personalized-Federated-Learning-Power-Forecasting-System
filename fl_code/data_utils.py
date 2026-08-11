"""Client data loader — load raw client data from processed CSVs.

No truncation or alignment — returns the full time range with leading/trailing
NaN preserved.  Each client's sequence-length distribution is reported.
"""

from __future__ import annotations

import yaml
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "fl_code" / "models" / "client_config.yaml"


def load_client_data(client_id: str) -> tuple[pd.DataFrame, dict]:
    """Load a client's raw data.

    1. Look up the client in ``client_config.yaml``.
    2. Load the corresponding processed CSV.
    3. Return the full time range (no truncation).  Leading / trailing NaN
       are preserved — they represent different start/end times across
       sequences.

    Parameters
    ----------
    client_id : str
        e.g. ``"lcl_res_0"``, ``"eld_ind_2"``, ``"steel_ind_0"``.

    Returns
    -------
    df : DataFrame
        Columns: datetime + load sequences + public features + local features.
    info : dict
        Metadata including per-sequence length distribution.
    """
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    dataset_id, client_cfg = _find_client(config, client_id)
    if dataset_id is None:
        raise KeyError(f"Client '{client_id}' not found in {CONFIG_PATH}")

    df = pd.read_csv(
        ROOT / "data" / "processed" / f"{dataset_id}.csv",
        parse_dates=["datetime"],
    )

    seqs = client_cfg["sequences"]
    public_cols = list(config[dataset_id]["public_features"])
    local_cols = list(config[dataset_id].get("local_features", []))

    # If config uses one-hot cat cols but CSV still has category_id,
    # temporarily load category_id for expansion
    _needs_onehot = (
        "cat_residential" in public_cols
        and "category_id" in df.columns
    )
    if _needs_onehot:
        csv_public = [c for c in public_cols if not c.startswith("cat_")] + ["category_id"]
    else:
        csv_public = public_cols

    keep_cols = ["datetime"] + seqs + csv_public + local_cols
    df = df[[c for c in keep_cols if c in df.columns]]

    # Expand category_id → one-hot
    if _needs_onehot:
        cat = df["category_id"].astype(int)
        df["cat_residential"] = (cat == 0).astype(float)
        df["cat_transformer"] = (cat == 1).astype(float)
        df["cat_industrial"]  = (cat == 2).astype(float)
        df = df.drop(columns=["category_id"])

    # Per-sequence length stats
    stats = _seq_length_stats(df, seqs)

    info = {
        "dataset": dataset_id,
        "client_id": client_id,
        "num_sequences": len(seqs),
        "total_rows": len(df),
        "time_range": (str(df["datetime"].iloc[0]), str(df["datetime"].iloc[-1])),
        "public_features": public_cols,
        "local_features": local_cols,
        **stats,
    }
    return df, info


def _seq_length_stats(df: pd.DataFrame, seqs: list[str]) -> dict:
    """Compute per-sequence length distribution."""
    valid_steps = {}
    first_valid_dt = {}
    last_valid_dt = {}
    for s in seqs:
        n = df[s].notna().sum()
        valid_steps[s] = int(n)
        first_idx = df[s].first_valid_index()
        last_idx = df[s].last_valid_index()
        first_valid_dt[s] = str(df.loc[first_idx, "datetime"]) if first_idx is not None else None
        last_valid_dt[s] = str(df.loc[last_idx, "datetime"]) if last_idx is not None else None

    steps = list(valid_steps.values())

    return {
        "valid_steps": {
            "min": int(np.min(steps)),
            "max": int(np.max(steps)),
            "mean": float(np.mean(steps)),
            "median": float(np.median(steps)),
            "p10": float(np.percentile(steps, 10)),
            "p90": float(np.percentile(steps, 90)),
        },
        "valid_days": {
            "min": np.min(steps) / 48,
            "max": np.max(steps) / 48,
            "mean": np.mean(steps) / 48,
        },
        "first_valid_range": (min(first_valid_dt.values()), max(first_valid_dt.values())),
        "last_valid_range": (min(last_valid_dt.values()), max(last_valid_dt.values())),
    }


def preprocess(df: pd.DataFrame, seqs: list[str],
               local_cols: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Normalise load sequences and local features.

    Load (power) columns — per-sequence:
        1. log1p  →  2. (y - mean) / std  over the valid (non-NaN) range.

    Local features — per-column:
        (x - mean) / std  over all rows where the column is valid.

    Parameters
    ----------
    df : DataFrame
        From :func:`load_client_data`.
    seqs : list[str]
        Load column names.
    local_cols : list[str] or None
        Local feature column names (can be empty list).

    Returns
    -------
    df_norm : DataFrame
        Normalised copy of *df* (datetime column unchanged).
    params : dict
        ``{column_name: {"log1p": bool, "mean": float, "std": float}}``.
        Use :func:`inverse_preprocess` to recover original values.
    """
    df_norm = df.copy()
    params = {}
    local_cols = local_cols or []

    # ---- load columns (per-sequence) ----
    for s in seqs:
        valid = df[s].notna()
        if valid.sum() == 0:
            params[s] = {"log1p": True, "mean": 0.0, "std": 1.0}
            continue
        df_norm[s] = df_norm[s].astype(float)
        y = df.loc[valid, s].values.astype(float)
        y_log = np.log1p(y)
        mu, sigma = y_log.mean(), y_log.std(ddof=0)
        if sigma < 1e-9:
            sigma = 1.0
        df_norm.loc[valid, s] = (y_log - mu) / sigma
        params[s] = {"log1p": True, "mean": float(mu), "std": float(sigma)}

    # ---- local feature columns (per-column) ----
    for c in local_cols:
        valid = df[c].notna()
        if valid.sum() == 0:
            params[c] = {"log1p": False, "mean": 0.0, "std": 1.0}
            continue
        df_norm[c] = df_norm[c].astype(float)
        x = df.loc[valid, c].values.astype(float)
        mu, sigma = x.mean(), x.std(ddof=0)
        if sigma < 1e-9:
            sigma = 1.0
        df_norm.loc[valid, c] = (x - mu) / sigma
        params[c] = {"log1p": False, "mean": float(mu), "std": float(sigma)}

    return df_norm, params


def inverse_preprocess(df_norm: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Reverse the normalisation applied by :func:`preprocess`."""
    df_out = df_norm.copy()
    for col, p in params.items():
        if col not in df_out.columns:
            continue
        valid = df_out[col].notna()
        if valid.sum() == 0:
            continue
        y = df_out.loc[valid, col].values.astype(float) * p["std"] + p["mean"]
        if p["log1p"]:
            y = np.expm1(y)
        df_out.loc[valid, col] = y
    return df_out


def _find_client(config: dict, client_id: str) -> tuple[str | None, dict | None]:
    for ds_id, ds_cfg in config.items():
        for cid, ccfg in ds_cfg["clients"].items():
            if cid == client_id:
                return ds_id, ccfg
    return None, None


# ============================================================================
# Train / test split
# ============================================================================

def split_train_test(df: pd.DataFrame, seqs: list[str],
                     train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-sequence chronological 80/20 split.

    Each sequence is split independently at ``train_ratio`` of its own valid
    (non-NaN) range.  The two returned DataFrames have the same shape and
    columns as ``df``:

    - **train_df**: test-portion values masked to NaN
    - **test_df**:  train-portion values masked to NaN

    Datetime and feature columns are preserved in both.

    Parameters
    ----------
    df : DataFrame
        From :func:`load_client_data`.
    seqs : list[str]
        Load column names.
    train_ratio : float
        Fraction of each sequence's valid steps to use for training.

    Returns
    -------
    train_df : DataFrame
    test_df : DataFrame
    """
    train_df = df.copy()
    test_df = df.copy()

    for s in seqs:
        f = df[s].first_valid_index()
        l = df[s].last_valid_index()
        if f is None or l is None:
            continue
        valid_len = l - f + 1
        split = f + int(valid_len * train_ratio)

        train_df.loc[split:l, s] = float("nan")
        test_df.loc[f:split - 1, s] = float("nan")

    return train_df, test_df


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python data_utils.py <client_id>")
        print("Example: python data_utils.py lcl_res_0")
        sys.exit(1)

    client_id = sys.argv[1]
    df, info = load_client_data(client_id)

    print(f"Client: {info['client_id']}")
    print(f"Dataset: {info['dataset']}")
    print(f"Total rows: {info['total_rows']}")
    print(f"Time range: {info['time_range'][0]} ~ {info['time_range'][1]}")
    print(f"Sequences: {info['num_sequences']}")
    print(f"Shape: {df.shape}")
    print(f"Public features: {info['public_features']}")
    print(f"Local features: {info['local_features']}")
    print()
    print("--- Valid steps per sequence ---")
    for k, v in info["valid_steps"].items():
        print(f"  {k}: {v:.0f}" if isinstance(v, float) else f"  {k}: {v}")
    print()
    print("--- Valid days per sequence ---")
    for k, v in info["valid_days"].items():
        print(f"  {k}: {v:.1f}")
    print()
    print(f"First valid range: {info['first_valid_range'][0]} ~ {info['first_valid_range'][1]}")
    print(f"Last valid range:  {info['last_valid_range'][0]} ~ {info['last_valid_range'][1]}")
    print()
    print(df.head(5).to_string())
    print()

    # Preprocessing
    seq_cols = [c for c in df.columns if c not in ("datetime",)
                and c not in info["public_features"]
                and c not in info["local_features"]]
    df_norm, norm_params = preprocess(df, seq_cols, info["local_features"])
    # Roundtrip check (only on non-NaN cells)
    df_rt = inverse_preprocess(df_norm, norm_params)
    rt_err = (df[seq_cols] - df_rt[seq_cols]).abs().max().max()

    print("--- Preprocessing ---")
    print(f"Roundtrip max error (should be ~0): {rt_err:.2e}")
    load_params = {k: v for k, v in norm_params.items() if v["log1p"]}
    print(f"Load sequences normalised: {len(load_params)} (log1p + mean/std)")
    feat_params = {k: v for k, v in norm_params.items() if not v["log1p"]}
    if feat_params:
        print(f"Local features normalised: {len(feat_params)} (mean/std)")
        for k, v in feat_params.items():
            print(f"  {k}: mean={v['mean']:.3f}, std={v['std']:.3f}")
    print()
    print("Normalised head:")
    print(df_norm.head(5).to_string())
