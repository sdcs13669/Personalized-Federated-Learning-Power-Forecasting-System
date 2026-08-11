#!/usr/bin/env python3
"""Generate client_config.yaml from processed CSVs + feature selection results.

Partition rules (v2.1 — by start time):
  - steel_ind: 1 client, whole dataset
  - tetouan_city: 3 clients (1 per zone)
  - lcl_res: 2 clients — start < 2012-01-01 / >= 2012-01-01
  - eld_ind: 3 clients — start < 2012 / 2012 / >= 2013

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
    df = pd.read_csv(PROC / "lcl_res.csv", low_memory=False, parse_dates=["datetime"])
    mac_cols = [c for c in df.columns if c.startswith("MAC")]
    # Partition by first valid datetime (smart meter enrollment wave)
    early = []   # start < 2012-01-01
    late = []    # start >= 2012-01-01
    for c in mac_cols:
        idx = df[c].first_valid_index()
        if idx is not None and df.loc[idx, "datetime"] < pd.Timestamp("2012-01-01"):
            early.append(c)
        else:
            late.append(c)
    config["lcl_res"] = {
        "public_features": PUBLIC_FEATURES,
        "local_features": LOCAL_FEATURES["lcl_res"],
        "clients": {
            "lcl_res_0": {
                "description": f"lcl_res early enrollment (<2012), {len(early)} MACs",
                "sequences": sorted(early),
            },
            "lcl_res_1": {
                "description": f"lcl_res late enrollment (>=2012), {len(late)} MACs",
                "sequences": sorted(late),
            },
        },
    }

    # ---- eld_ind ----
    df = pd.read_csv(PROC / "eld_ind.csv", low_memory=False, parse_dates=["datetime"])
    mt_cols = [c for c in df.columns if c.startswith("MT_")]
    # Partition by first valid datetime (transformer commissioning year)
    g1 = []   # start < 2012-01-01
    g2 = []   # 2012-01-01 <= start < 2013-01-01
    g3 = []   # start >= 2013-01-01
    for c in mt_cols:
        idx = df[c].first_valid_index()
        if idx is None:
            continue
        start = df.loc[idx, "datetime"]
        if start < pd.Timestamp("2012-01-01"):
            g1.append(c)
        elif start < pd.Timestamp("2013-01-01"):
            g2.append(c)
        else:
            g3.append(c)
    config["eld_ind"] = {
        "public_features": PUBLIC_FEATURES,
        "local_features": LOCAL_FEATURES["eld_ind"],
        "clients": {
            "eld_ind_0": {
                "description": f"eld_ind commissioned before 2012, {len(g1)} MTs",
                "sequences": sorted(g1),
            },
            "eld_ind_1": {
                "description": f"eld_ind commissioned in 2012, {len(g2)} MTs",
                "sequences": sorted(g2),
            },
            "eld_ind_2": {
                "description": f"eld_ind commissioned >=2013, {len(g3)} MTs",
                "sequences": sorted(g3),
            },
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
