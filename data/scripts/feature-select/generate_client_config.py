#!/usr/bin/env python3
"""Generate client_config.yaml from processed CSVs + feature selection results.

Partition rules (v1.2):
  - steel_ind: 1 client, whole dataset
  - tetouan_city: 3 clients (1 per zone)
  - lcl_res: 2 clients — <730 days / >=730 days
  - eld_ind: drop <365 days, then 3 clients — [365,730) / [730,1095) / >=1095

Output: fl_code/models/client_config.yaml
"""
from __future__ import annotations

import yaml
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "fl_code" / "models" / "client_config.yaml"

PUBLIC_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "month_sin", "month_cos", "category_id",
]

# Feature selection results (from SHAP reports)
LOCAL_FEATURES = {
    "eld_ind": [],
    "lcl_res": [],
    "steel_ind": ["co2", "lagging_reactive_power", "lagging_pf", "load_type"],
    "tetouan_city": ["temperature", "humidity", "wind_speed",
                     "general_diffuse_flow", "diffuse_flow"],
}


def _valid_days(series: pd.Series) -> float:
    """Number of days with valid data (30min steps)."""
    return series.notna().sum() * 0.5 / 24


def build_config() -> dict:
    config = {}

    # ---- steel_ind ----
    df = pd.read_csv(PROC / "steel_ind.csv", low_memory=False)
    config["steel_ind"] = {
        "public_features": PUBLIC_FEATURES,
        "local_features": LOCAL_FEATURES["steel_ind"],
        "clients": {
            "steel_ind_0": {
                "description": "steel_ind full dataset (365 days)",
                "sequences": ["Usage_kWh"],
            },
        },
    }

    # ---- tetouan_city ----
    df = pd.read_csv(PROC / "tetouan_city.csv", low_memory=False)
    zones = [c for c in df.columns if c.startswith("load_zone")]
    clients = {}
    for i, z in enumerate(sorted(zones)):
        zone_desc = {0: "Zone1 (industrial)", 1: "Zone2 (mixed)", 2: "Zone3 (residential)"}
        clients[f"tetouan_city_{i}"] = {
            "description": f"tetouan_city {zone_desc.get(i, z)} (364 days)",
            "sequences": [z],
        }
    config["tetouan_city"] = {
        "public_features": PUBLIC_FEATURES,
        "local_features": LOCAL_FEATURES["tetouan_city"],
        "clients": clients,
    }

    # ---- lcl_res ----
    df = pd.read_csv(PROC / "lcl_res.csv", low_memory=False)
    mac_cols = [c for c in df.columns if c.startswith("MAC")]
    days = {c: _valid_days(df[c]) for c in mac_cols}
    short = sorted([c for c, d in days.items() if d < 730])
    long = sorted([c for c, d in days.items() if d >= 730])
    config["lcl_res"] = {
        "public_features": PUBLIC_FEATURES,
        "local_features": LOCAL_FEATURES["lcl_res"],
        "clients": {
            "lcl_res_0": {
                "description": f"lcl_res short sequences (<730 days), {len(short)} MACs",
                "sequences": short,
            },
            "lcl_res_1": {
                "description": f"lcl_res long sequences (>=730 days), {len(long)} MACs",
                "sequences": long,
            },
        },
    }

    # ---- eld_ind ----
    df = pd.read_csv(PROC / "eld_ind.csv", low_memory=False)
    mt_cols = [c for c in df.columns if c.startswith("MT_")]
    days = {c: _valid_days(df[c]) for c in mt_cols}
    # Filter <365 days
    valid = {c: d for c, d in days.items() if d >= 365}
    dropped = sorted([c for c, d in days.items() if d < 365])
    g1 = sorted([c for c, d in valid.items() if d < 730])       # [365, 730)
    g2 = sorted([c for c, d in valid.items() if 730 <= d < 1095])  # [730, 1095)
    g3 = sorted([c for c, d in valid.items() if d >= 1095])      # >= 1095
    config["eld_ind"] = {
        "public_features": PUBLIC_FEATURES,
        "local_features": LOCAL_FEATURES["eld_ind"],
        "clients": {
            "eld_ind_0": {
                "description": f"eld_ind [365,730) days, {len(g1)} MTs",
                "sequences": g1,
            },
            "eld_ind_1": {
                "description": f"eld_ind [730,1095) days, {len(g2)} MTs",
                "sequences": g2,
            },
            "eld_ind_2": {
                "description": f"eld_ind >=1095 days, {len(g3)} MTs",
                "sequences": g3,
            },
        },
        "dropped_sequences": {
            "reason": "valid days < 365",
            "count": len(dropped),
            "sequences": dropped,
        },
    }

    return config


def main():
    config = build_config()

    # Summary
    total_clients = sum(len(v["clients"]) for v in config.values())
    print(f"Total clients: {total_clients}")
    for ds, cfg in config.items():
        n = len(cfg["clients"])
        seqs = [len(c["sequences"]) for c in cfg["clients"].values()]
        print(f"  {ds}: {n} clients, features: "
              f"{len(cfg['public_features'])} public + {len(cfg['local_features'])} local, "
              f"sequences per client: {seqs}")

    with open(OUT, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
