#!/usr/bin/env python3
"""Phase 2: Per-dataset Global TCN baselines (4 datasets × 2 feature modes).

Two baseline types:
  --features public : 8 time encodings + historical load = 9 channels
  --features all    : public + local features = 9+D_local channels

Usage:
  python fl_code/train_baseline.py --features public --epochs 1 --datasets steel_ind
  python fl_code/train_baseline.py --features all --epochs 30
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fl_code"))

from models import FedTCNConfig, build_global_model  # noqa: E402
from data_utils import (  # noqa: E402
    INPUT_STEPS, PRED_LEN, WINDOW, DEFAULT_STRIDE, TRAIN_RATIO,
    FEATURE_CONFIG,
    SequenceData,
    load_client_config,
    load_dataset_df,
    build_sequence_public,
    build_sequence_all,
    split_chronological,
    split_by_sequence,
    WindowDataset,
)

DATA_DIR = ROOT / "data" / "processed"
CONFIG_PATH = ROOT / "fl_code" / "models" / "client_config.yaml"
OUT_DIR = ROOT / "fl_code" / "outputs" / "baseline_global_tcn"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
RESULT_DIR = OUT_DIR / "results"

ALL_DATASETS = ["steel_ind", "tetouan_city", "lcl_res", "eld_ind"]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion,
                device, pbar, clip_norm: float = 5.0) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.update(1)
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Rolling forecast for visualisation
# ---------------------------------------------------------------------------
@torch.no_grad()
def rolling_forecast(model, seq: SequenceData, pub: np.ndarray,
                     start: int, n_days: int, local_feat_names,
                     device) -> tuple[np.ndarray, np.ndarray]:
    """Roll forward n_days, each step = PRED_LEN (1 day). Returns (y_pred, y_true)
    in ORIGINAL scale (expm1'd), shape (n_days * PRED_LEN,)."""
    preds = []
    trues = []
    cur = start
    for _ in range(n_days):
        # build input window [cur:cur+INPUT_STEPS]
        pub_slice = pub[cur:cur + INPUT_STEPS]                     # (336, 8)
        load_slice = seq.load_z[cur:cur + INPUT_STEPS]             # (336,)
        load_slice = np.nan_to_num(load_slice, nan=0.0)
        channels = [pub_slice, load_slice.reshape(-1, 1)]
        for fn in local_feat_names:
            feat = seq.local_feats.get(fn)
            if feat is not None:
                sl = feat[cur:cur + INPUT_STEPS]
                channels.append(np.nan_to_num(sl, nan=0.0).reshape(-1, 1))
            else:
                channels.append(np.zeros((INPUT_STEPS, 1), dtype=np.float32))
        x = np.concatenate(channels, axis=1).T
        x_t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(device)

        y_pred_log = model(x_t).cpu().numpy().squeeze(0)           # (48,)
        y_true_log = seq.y_log[cur + INPUT_STEPS:cur + INPUT_STEPS + PRED_LEN]

        preds.append(np.expm1(y_pred_log))
        trues.append(np.expm1(y_true_log))

        cur += PRED_LEN  # advance by 1 day (true values)

    return np.concatenate(preds), np.concatenate(trues)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, test_seqs: list[SequenceData], pub: np.ndarray,
             local_feat_names: list[str], device, batch_size: int) -> dict:
    """Evaluate on test windows. Returns per-sequence MAE + dataset-level stats."""
    model.eval()
    all_mae = []
    sum_abs = 0.0
    sum_sq = 0.0
    n_steps = 0

    for s in test_seqs:
        s_abs = 0.0
        s_sq = 0.0
        s_n = 0
        for i in range(0, len(s.test_starts), batch_size):
            batch = s.test_starts[i:i + batch_size]
            # build batch manually (same as WindowDataset but for test starts)
            x_list, y_list = [], []
            for st in batch:
                pub_sl = pub[st:st + INPUT_STEPS]
                load_sl = np.nan_to_num(s.load_z[st:st + INPUT_STEPS], nan=0.0)
                ch = [pub_sl, load_sl.reshape(-1, 1)]
                for fn in local_feat_names:
                    f = s.local_feats.get(fn)
                    ch.append(np.nan_to_num(f[st:st + INPUT_STEPS], nan=0.0).reshape(-1, 1)
                              if f is not None else np.zeros((INPUT_STEPS, 1), dtype=np.float32))
                x_list.append(np.concatenate(ch, axis=1).T)
                y_list.append(s.y_log[st + INPUT_STEPS:st + WINDOW])
            x_t = torch.from_numpy(np.ascontiguousarray(np.stack(x_list))).to(device)
            y_pred_log = model(x_t).cpu().numpy()
            y_true_log = np.stack(y_list)

            y_pred = np.expm1(y_pred_log)
            y_true = np.expm1(y_true_log)
            err = y_pred - y_true
            s_abs += np.abs(err).sum()
            s_sq += (err ** 2).sum()
            s_n += err.size

        if s_n > 0:
            all_mae.append((s_abs / s_n, s.name))
        sum_abs += s_abs
        sum_sq += s_sq
        n_steps += s_n

    mae = sum_abs / n_steps if n_steps > 0 else float("nan")
    mse = sum_sq / n_steps if n_steps > 0 else float("nan")
    rmse = math.sqrt(mse)
    return {"mae": round(float(mae), 6), "mse": round(float(mse), 6),
            "rmse": round(float(rmse), 6), "n_steps": n_steps,
            "seq_mae": sorted(all_mae, key=lambda x: x[0])}


# ---------------------------------------------------------------------------
# Train one dataset
# ---------------------------------------------------------------------------
def train_one_dataset(ds_id: str, ds_cfg: dict, pub_cols: list[str],
                      args) -> dict:
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_id}  |  Features: {args.features}")
    print(f"{'='*60}")

    # ---- Collect sequence & local-feature column names ----
    all_seq_cols: list[str] = []
    for cl in ds_cfg["clients"].values():
        all_seq_cols.extend(cl["sequences"])

    local_cols: list[str] = []
    fc = FEATURE_CONFIG.get(ds_id, {})
    local_cols = fc.get("unbounded", []) + fc.get("bounded", [])

    # ---- Load CSV ----
    df = load_dataset_df(DATA_DIR / f"{ds_id}.csv", all_seq_cols, pub_cols, local_cols)
    pub = df[pub_cols].to_numpy(dtype=np.float32)

    # ---- Build sequences ----
    build_fn = build_sequence_all if args.features == "all" else build_sequence_public
    seqs: list[SequenceData] = []
    n_skip = 0
    for col in all_seq_cols:
        if args.features == "all":
            sd = build_sequence_all(df[col], col, ds_id, df, args.stride)
        else:
            sd = build_sequence_public(df[col], col, args.stride)
        if sd is not None:
            seqs.append(sd)
        else:
            n_skip += 1
    if n_skip:
        print(f"  Skipped {n_skip} sequences (too short)")

    # ---- Split ----
    if ds_id in ("steel_ind", "tetouan_city"):
        train_seqs, test_seqs = split_chronological(seqs)
    else:
        train_seqs, test_seqs = split_by_sequence(seqs, seed=args.seed)

    n_train = sum(len(s.train_starts) for s in train_seqs)
    n_test = sum(len(s.test_starts) for s in test_seqs)
    print(f"  {len(train_seqs)} train seqs ({n_train} windows), "
          f"{len(test_seqs)} test seqs ({n_test} windows)")

    if n_train == 0:
        print(f"  SKIP: no train windows")
        return None

    # ---- Determine local feature names for channel ordering ----
    local_feat_names = []
    if args.features == "all":
        local_feat_names = fc.get("unbounded", []) + fc.get("bounded", [])
    in_channels = 9 + len(local_feat_names)

    # ---- Model ----
    model_cfg = FedTCNConfig()
    model_cfg.global_model.in_channels = in_channels
    model_cfg.global_model.input_steps = INPUT_STEPS
    model_cfg.global_model.pred_len = PRED_LEN
    model_cfg.input_window_steps = INPUT_STEPS
    model_cfg.output_window_steps = PRED_LEN
    model = build_global_model(model_cfg).to(args.device)

    # Initialise final decoder bias to training-set mean of y_log so the
    # model starts by predicting the mean (not 0).  Without this the decoder
    # needs many epochs just to shift its output scale.
    all_y_log_train = np.concatenate(
        [s.y_log[s.train_starts + INPUT_STEPS][:PRED_LEN] for s in train_seqs
         if len(s.train_starts) > 0])
    if len(all_y_log_train) > 0:
        mean_y = float(np.nanmean(all_y_log_train))
        # Last layer of decoder is a Linear; set its bias.
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.out_features == PRED_LEN:
                with torch.no_grad():
                    m.bias.copy_(torch.full((PRED_LEN,), mean_y))
                break

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: GlobalTCN in={in_channels} pred={PRED_LEN}, {n_params:,} params")

    # ---- DataLoader (adaptive batch size for small datasets) ----
    bs = min(args.batch_size, max(4, n_train // 10))
    if bs != args.batch_size:
        print(f"  batch_size adjusted: {args.batch_size} -> {bs} ({n_train} windows)")
    train_ds = WindowDataset(train_seqs, pub, in_channels, local_feat_names)
    train_loader = DataLoader(train_ds, batch_size=bs,
                              shuffle=True, num_workers=0,
                              pin_memory=(args.device.type == "cuda"))

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-5)

    # ---- Training ----
    print(f"  Training: {args.epochs} epochs, batch={bs}, lr={args.lr}")
    loss_history = []
    t0 = time.time()
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc=f"  {ds_id}",
                      unit="epoch", ncols=100)
    for epoch in epoch_pbar:
        batch_pbar = tqdm(total=len(train_loader), desc=f"    Epoch {epoch}",
                          unit="batch", leave=False, ncols=90)
        avg_loss = train_epoch(model, train_loader, optimizer, criterion,
                               args.device, batch_pbar)
        loss_history.append(round(avg_loss, 6))
        batch_pbar.close()
        epoch_pbar.set_postfix(loss=f"{avg_loss:.6f}")
    epoch_pbar.close()
    train_sec = time.time() - t0
    print(f"  Training done in {train_sec:.0f}s")

    # ---- Evaluation ----
    print("  Evaluating...")
    eval_result = evaluate(model, test_seqs, pub, local_feat_names,
                           args.device, args.batch_size)
    print(f"    MAE={eval_result['mae']:.4f}  MSE={eval_result['mse']:.4f}  "
          f"RMSE={eval_result['rmse']:.4f}")

    # ---- Visualisation: best / median / worst test sequences (rolling 14-day) ----
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    seq_mae_list = eval_result["seq_mae"]  # sorted by MAE ascending
    if seq_mae_list:
        picks = [
            ("best", seq_mae_list[0]),
            ("median", seq_mae_list[len(seq_mae_list) // 2]),
            ("worst", seq_mae_list[-1]),
        ]
        for label, (mae_val, sname) in picks:
            s = next((x for x in test_seqs if x.name == sname), None)
            if s is None or len(s.test_starts) == 0:
                continue
            start0 = int(s.test_starts[0])
            y_pred, y_true = rolling_forecast(
                model, s, pub, start0, n_days=14,
                local_feat_names=local_feat_names, device=args.device)
            # Plot: last ~7 days of input + 14 days forecast
            hist_steps = 336  # 7 days
            hist_start = max(0, start0)
            hist_y = np.expm1(s.y_log[hist_start:start0 + INPUT_STEPS])
            hist_x = np.arange(-len(hist_y), 0) * 0.5 / 24  # days
            pred_x = np.arange(0, len(y_pred)) * 0.5 / 24

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(hist_x[-hist_steps:], hist_y[-hist_steps:],
                    color="gray", alpha=0.7, label="history")
            ax.plot(pred_x, y_true, color="blue", linewidth=1.5, label="true")
            ax.plot(pred_x, y_pred, color="red", linewidth=1.5, label="pred")
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.set_xlabel("days from forecast start")
            ax.set_ylabel("load")
            ax.set_title(f"{ds_id}/{args.features} — {label} ({sname})  MAE={mae_val:.4f}")
            ax.legend()
            fig.tight_layout()
            fig_path = FIG_DIR / f"{ds_id}_{args.features}_{label}_{sname}.png"
            fig.savefig(fig_path)
            plt.close(fig)
        print(f"  Saved figures to {FIG_DIR}")

    # ---- Save model ----
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{ds_id}_{args.features}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model_cfg.to_dict(),
        "hyperparams": {"epochs": args.epochs, "batch_size": args.batch_size,
                        "lr": args.lr, "loss": "mae"},
        "seed": args.seed,
        "in_channels": in_channels,
        "n_params": n_params,
    }, model_path)
    print(f"  Saved {model_path}")

    return {
        "features": args.features,
        "in_channels": in_channels,
        "n_train_windows": n_train,
        "n_test_windows": n_test,
        "train_seconds": round(train_sec, 1),
        "n_params": n_params,
        "train_loss_history": loss_history,
        "test": {"mae": eval_result["mae"], "mse": eval_result["mse"],
                 "rmse": eval_result["rmse"]},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Phase 2: Per-dataset Global TCN baselines")
    ap.add_argument("--features", choices=["public", "all"], default="public",
                    help="public=time+history (9ch), all=+local features")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--stride", type=int, default=6,
                    help="sample Window stride in 30min steps (default 6 = 3 hours)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--datasets", default="all",
                    help="Comma-separated dataset ids, or 'all'")
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    seed_everything(args.seed)

    args.device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {args.device}  |  Features: {args.features}")

    datasets = ALL_DATASETS if args.datasets == "all" else \
        [d.strip() for d in args.datasets.split(",")]

    cfg = load_client_config(args.config)
    pub_cols = cfg["steel_ind"]["public_features"]

    # ---- Run each dataset ----
    all_results = {}
    for ds_id in datasets:
        if ds_id not in cfg:
            print(f"WARNING: {ds_id} not in config, skipping")
            continue
        result = train_one_dataset(ds_id, cfg[ds_id], pub_cols, args)
        if result is not None:
            all_results[ds_id] = result

    # ---- Write results.json ----
    output = {
        "meta": {"script": "fl_code/train_baseline.py",
                 "created_at": datetime.now().isoformat(),
                 "seed": args.seed, "device": str(args.device),
                 "features": args.features},
        "hyperparams": {"epochs": args.epochs, "batch_size": args.batch_size,
                        "lr": args.lr, "loss": "mae", "optimizer": "adam",
                        "weight_decay": 1e-5,
                        "window": {"input": INPUT_STEPS, "pred": PRED_LEN,
                                   "stride": args.stride}},
        "results": all_results,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULT_DIR / f"{args.features}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()
