"""Client data loader — load raw client data from processed CSVs.

No truncation or alignment — returns the full time range with leading/trailing
NaN preserved.  Each client's sequence-length distribution is reported.
"""

from __future__ import annotations

import yaml
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
    public_cols = config[dataset_id]["public_features"]
    local_cols = config[dataset_id].get("local_features", [])

    keep_cols = ["datetime"] + seqs + public_cols + local_cols
    df = df[keep_cols]

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
    import numpy as np

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


def _find_client(config: dict, client_id: str) -> tuple[str | None, dict | None]:
    for ds_id, ds_cfg in config.items():
        for cid, ccfg in ds_cfg["clients"].items():
            if cid == client_id:
                return ds_id, ccfg
    return None, None


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
