"""Phase 2: Federated Baseline — FedAvg GlobalTCN training.

Single-process FedAvg simulation across all clients on one machine (no
federated-learning library).  Trains a Global TCN point-forecast model (no
ResidualCorrector) to produce the ``Y_pre`` baseline.  This serves as the
control against which Phase 3 personalization gains are measured.

DP mode (``--dp-noise``): each client clips its own update and adds Gaussian
noise **locally before upload**; the server only aggregates — it never sees
an un-noised update.

Usage::

    python -m fl_code.train_baseline                              # defaults
    python -m fl_code.train_baseline --rounds 30 --lr 0.001       # custom
    python -m fl_code.train_baseline --clients steel_ind_0 eld_ind_0  # subset
    python -m fl_code.train_baseline --max-seqs 5 --eval-seqs 3   # fast dev
    python -m fl_code.train_baseline --output-dir my_run            # custom output root
"""

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from fl_code.data_utils import (
    load_client_data,
    preprocess,
    LazySlidingWindowDataset,
)
from fl_code.train_eval_utils import train_epoch, evaluate
from fl_code.models import TCNConfig, build_tcn
from fl_code.config import (
    INPUT_STEPS, PRED_LEN, STRIDE, TRAIN_RATIO,
    BASELINE_ROUNDS, BASELINE_LOCAL_EPOCHS,
    BASELINE_LR, BASELINE_BATCH_SIZE,
)

ROOT = Path(__file__).resolve().parents[1]
CLIENT_CONFIG_PATH = ROOT / "fl_code" / "models" / "client_config.yaml"


# ============================================================================
# FedAvg round (single process)
# ============================================================================

def _train_client(model: torch.nn.Module, train_ds: LazySlidingWindowDataset,
                  lr: float, batch_size: int, local_epochs: int,
                  device: str) -> float:
    """Train ``model`` locally for ``local_epochs``; return last epoch's MAE loss."""
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = float("nan")
    for _ in range(local_epochs):
        loss = train_epoch(model, loader, optimizer, device)
    return loss


def _clip_delta_inplace(delta: OrderedDict[str, torch.Tensor],
                        clipping_norm: float) -> float:
    """In-place l2 clip over all tensors concatenated; return pre-clip norm."""
    norm = float(np.sqrt(sum(t.double().pow(2).sum().item() for t in delta.values())))
    scale = min(1.0, clipping_norm / max(norm, 1e-6))
    for t in delta.values():
        t.mul_(scale)
    return norm


def _add_noise_inplace(delta: OrderedDict[str, torch.Tensor],
                       noise_std: float) -> None:
    """Add independent per-tensor Gaussian noise N(0, noise_std²)."""
    for t in delta.values():
        t.add_(torch.randn_like(t) * noise_std)


def _fedavg_round(global_state: OrderedDict[str, torch.Tensor],
                  cache: dict, weights: dict, args: argparse.Namespace,
                  device: str, dp: dict | None
                  ) -> tuple[OrderedDict[str, torch.Tensor], dict[str, float]]:
    """One FedAvg round: local train per client → Δᵢ → (DP: local clip+noise)
    → weighted aggregate.  Returns (new_global_state, {cid: train_loss})."""
    state_keys = list(global_state.keys())
    agg = OrderedDict((k, torch.zeros_like(global_state[k])) for k in state_keys)
    losses: dict[str, float] = {}

    for cid in cache:
        model = build_tcn(TCNConfig()).to(device)
        model.load_state_dict(global_state)
        losses[cid] = _train_client(model, cache[cid]["train_ds"], args.lr,
                                    args.batch_size, args.local_epochs, device)
        delta = OrderedDict(
            (k, model.state_dict()[k] - global_state[k]) for k in state_keys)
        del model
        if dp:
            # Client-side DP: clip + noise locally before the update leaves
            # the client — the server never sees an un-noised update.
            _clip_delta_inplace(delta, dp["clipping_norm"])
            _add_noise_inplace(delta,
                               dp["noise_multiplier"] * dp["clipping_norm"])
        w = weights[cid]
        for k in state_keys:
            agg[k].add_(delta[k], alpha=w)

    new_state = OrderedDict((k, global_state[k] + agg[k]) for k in state_keys)
    return new_state, losses


# ============================================================================
# Data helpers
# ============================================================================

def _load_client_cache(client_id: str, stride: int,
                       max_seqs: int | None = None) -> dict:
    """Load + preprocess client data; returned dict holds the train Dataset
    plus the metadata needed for post-training eval."""
    df, info = load_client_data(client_id)

    feat_names = set(info["public_features"] + info["local_features"])
    seqs = [c for c in df.columns if c not in feat_names and c != "datetime"]

    if max_seqs and len(seqs) > max_seqs:
        seqs = seqs[:max_seqs]

    df_norm, _ = preprocess(df, seqs, info["local_features"])

    train_ds = LazySlidingWindowDataset(
        df_norm, seqs, info["public_features"],
        stride=stride, train=True,
    )

    return {
        "df_norm": df_norm,
        "seqs": seqs,
        "public_cols": info["public_features"],
        "local_cols": info["local_features"],
        "train_ds": train_ds,
        "n_train": len(train_ds),
    }


def _compute_epsilon(noise_multiplier: float, rounds: int,
                     sample_rate: float, delta: float) -> float:
    """(ε, δ)-DP budget for DP-FedAvg via RDP composition (Opacus accountant).

    ``sample_rate = num_sampled_clients / num_clients`` — every client
    participates in every round, so it is 1.0.
    """
    from opacus.accountants import RDPAccountant
    accountant = RDPAccountant()
    for _ in range(rounds):
        accountant.step(noise_multiplier=noise_multiplier,
                        sample_rate=sample_rate)
    return float(accountant.get_epsilon(delta))


def _list_clients(whitelist: list[str] | None = None) -> list[str]:
    """All client IDs from client_config.yaml, optionally filtered."""
    with open(CLIENT_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    ids: list[str] = []
    for ds_cfg in config.values():
        for cid in ds_cfg["clients"]:
            ids.append(cid)

    if whitelist:
        ids = [c for c in ids if c in whitelist]
    return ids


# ============================================================================
# Post-training evaluation (rolling forecast)
# ============================================================================

def _eval_all_clients(model: torch.nn.Module, cache: dict, args: argparse.Namespace,
                      device: str) -> dict:
    """Run rolling-forecast evaluate() on every client, return aggregate metrics."""
    all_preds = []
    all_actuals = []
    per_client = {}

    for cid, data in cache.items():
        eval_seqs = data["seqs"]
        if args.eval_seqs and len(eval_seqs) > args.eval_seqs:
            eval_seqs = eval_seqs[:args.eval_seqs]

        metrics = evaluate(
            model, data["df_norm"], eval_seqs, data["public_cols"],
            stride=args.stride, device=device,
        )
        per_client[cid] = {
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "n_train": data["n_train"],
        }
        if metrics["predictions"] is not None:
            all_preds.append(metrics["predictions"])
            all_actuals.append(metrics["actuals"])

    wape = float("nan")
    avg_mae = float("nan")
    if all_preds:
        all_p = np.concatenate(all_preds)
        all_a = np.concatenate(all_actuals)
        wape = float(np.sum(np.abs(all_p - all_a)) / np.sum(np.abs(all_a)))
        valid_maes = [m["mae"] for m in per_client.values()
                      if not np.isnan(m["mae"])]
        avg_mae = float(np.mean(valid_maes)) if valid_maes else float("nan")

    return {"wape": wape, "avg_mae": avg_mae, "client_metrics": per_client}


# ============================================================================
# Main
# ============================================================================

def main(args: argparse.Namespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    client_ids = _list_clients(args.clients)
    print(f"Clients ({len(client_ids)}): {', '.join(client_ids)}")

    # --- Pre-load all client data ---
    print("Loading client data ...")
    cache: dict[str, dict] = {}
    total_windows = 0
    for cid in client_ids:
        cache[cid] = _load_client_cache(cid, args.stride, args.max_seqs)
        n = cache[cid]["n_train"]
        total_windows += n
        print(f"  {cid}: {n} training windows")
    print(f"Total training windows: {total_windows:,}")

    # --- Build initial model parameters ---
    tmp_model = build_tcn(TCNConfig())
    n_params = sum(p.numel() for p in tmp_model.parameters())
    print(f"Model params: {n_params:,}")

    # DP-FedAvg when --dp-noise given.  Privacy is enforced **client-side**:
    # each client clips its own update to l2 ≤ C and adds Gaussian noise
    # N(0, (σC)²) before uploading — the server (aggregator) never sees an
    # un-noised update, so the privacy guarantee does not depend on trusting
    # the server.  Per-client mechanism: sensitivity C, noise σC → noise
    # multiplier σ, aggregated with any weights (post-processing).
    dp_info = None
    if args.dp_noise is not None and args.dp_noise > 0:
        sample_rate = 1.0   # every client participates in every round
        eps = _compute_epsilon(args.dp_noise, args.rounds,
                               sample_rate, args.dp_delta)
        dp_info = {
            "noise_multiplier": args.dp_noise,
            "clipping_norm": args.dp_clip,
            "delta": args.dp_delta,
            "epsilon": round(eps, 4),
        }
        print(f"DP-FedAvg enabled (client-side clipping + noise): "
              f"σ={args.dp_noise}, C={args.dp_clip}, "
              f"(ε={eps:.2f}, δ={args.dp_delta}) after {args.rounds} rounds")

    # DP and non-DP runs write to separate sub-directories so they never
    # overwrite each other.
    variant_dir = Path(args.output_dir) / ("dp" if dp_info else "nodp")
    variant_dir.mkdir(parents=True, exist_ok=True)

    # --- Save run config (model architecture + hyperparameters) ---
    tcn_cfg = TCNConfig().to_dict()
    rf = 1 + 2 * (tcn_cfg["kernel_size"] - 1) * (2 ** len(tcn_cfg["num_channels"]) - 1)
    config_json = {
        "script": "fl_code.train_baseline",
        "phase": "Phase 2 — FedAvg GlobalTCN (single-process simulation)",
        "model": {
            "name": "GlobalTCN",
            **tcn_cfg,
            "receptive_field": rf,
            "n_params": n_params,
        },
        "window_geometry": {
            "input_steps": INPUT_STEPS,
            "pred_len": PRED_LEN,
            "stride": args.stride,
            "train_ratio": TRAIN_RATIO,
        },
        "training": {
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "num_clients": len(client_ids),
            "clients": client_ids,
            "max_seqs": args.max_seqs,
            "eval_seqs": args.eval_seqs,
            "loss": "MAE (L1) on normalised load",
            "checkpoints": "one per round (checkpoints/round_NNN.pt); final = last round",
        },
        "dp": dp_info,
    }
    with open(variant_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2, default=str)
    print(f"Saved run config to {variant_dir / 'config.json'}")

    # --- Run FedAvg rounds (single process) ---
    print(f"\nStarting FedAvg simulation: {args.rounds} rounds × "
          f"{len(client_ids)} clients")

    state_keys = list(tmp_model.state_dict().keys())
    global_state = OrderedDict(
        (k, v.to(device).clone()) for k, v in tmp_model.state_dict().items())
    weights = {cid: cache[cid]["n_train"] / total_windows for cid in client_ids}
    per_client_losses: dict[int, dict[str, float]] = {}
    ckpt_dir = variant_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    for r in range(1, args.rounds + 1):
        global_state, losses = _fedavg_round(global_state, cache, weights,
                                             args, device, dp_info)
        per_client_losses[r] = losses
        mean_loss = float(np.mean(list(losses.values())))
        print(f"Round {r:3d}/{args.rounds}  train_loss={mean_loss:.6f}")
        torch.save(OrderedDict((k, v.detach().cpu())
                               for k, v in global_state.items()),
                   ckpt_dir / f"round_{r:03d}.pt")
    elapsed = time.perf_counter() - t0
    print(f"Training complete in {elapsed:.0f}s")
    print(f"Saved {args.rounds} round checkpoints to {ckpt_dir}")

    # --- Load final aggregated model for eval ---
    tmp_model.load_state_dict(
        OrderedDict((k, v.detach().cpu()) for k, v in global_state.items()))

    # --- Post-training evaluation ---
    print("\nEvaluating final model on all clients ...")
    eval_device = device
    tmp_model.to(eval_device)
    tmp_model.eval()

    results = _eval_all_clients(tmp_model, cache, args, eval_device)

    print(f"\nFinal — WAPE={results['wape']:.4f}  MAE={results['avg_mae']:.4f}")

    # --- Save outputs (final model = last round checkpoint) ---
    with open(variant_dir / "baseline_history.json", "w") as f:
        json.dump({
            "args": {k: str(v) for k, v in vars(args).items()},
            "num_clients": len(client_ids),
            "total_train_windows": total_windows,
            "model_params": n_params,
            "training_time_s": round(elapsed, 1),
            "final_metrics": results,
            "train_losses": [round(float(np.mean(list(per_client_losses[r].values()))), 6)
                             for r in range(1, args.rounds + 1)],
            "train_losses_per_client": per_client_losses,
            "final_model": f"checkpoints/round_{args.rounds:03d}.pt",
            "dp": dp_info,   # None unless --dp-noise was given
        }, f, indent=2, default=str)

    # --- Per-client summary ---
    print("\n--- Per-client final metrics ---")
    for cid in client_ids:
        m = results["client_metrics"].get(cid, {})
        print(f"  {cid:20s}  MAE={m.get('mae', float('nan')):.4f}  "
              f"RMSE={m.get('rmse', float('nan')):.4f}  n_train={m.get('n_train', 0)}")

    print(f"\nOutputs saved to {variant_dir}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: FedAvg GlobalTCN Baseline Training "
                    "(single-process FedAvg simulation)",
    )
    parser.add_argument("--rounds", type=int, default=BASELINE_ROUNDS,
                        help=f"Communication rounds (default: {BASELINE_ROUNDS})")
    parser.add_argument("--local-epochs", type=int,
                        default=BASELINE_LOCAL_EPOCHS,
                        help=f"Local SGD epochs per round (default: {BASELINE_LOCAL_EPOCHS})")
    parser.add_argument("--lr", type=float, default=BASELINE_LR,
                        help=f"Learning rate (default: {BASELINE_LR})")
    parser.add_argument("--batch-size", type=int, default=BASELINE_BATCH_SIZE,
                        help=f"Batch size (default: {BASELINE_BATCH_SIZE})")
    parser.add_argument("--stride", type=int, default=STRIDE,
                        help=f"Sliding-window stride (default: {STRIDE}, "
                             f"= pred_len for continuous coverage)")
    parser.add_argument("--eval-seqs", type=int, default=None,
                        help="Cap eval to first N sequences per client (default: all)")
    parser.add_argument("--max-seqs", type=int, default=None,
                        help="Cap training sequences per client (default: all)")
    parser.add_argument("--clients", nargs="*", default=None,
                        help="Client ids to include (default: all)")
    parser.add_argument("--output-dir", type=str,
                        default=str(ROOT / "fl_code" / "baseline_outputs"),
                        help="Output root directory; DP runs go to "
                             "<output-dir>/dp, non-DP to <output-dir>/nodp "
                             "(default: fl_code/baseline_outputs)")

    # DP-FedAvg — client-side clipping + Gaussian noise before upload
    parser.add_argument("--dp-noise", type=float, default=None,
                        help="Enable DP-FedAvg: Gaussian noise multiplier σ "
                             "(default: None = no DP)")
    parser.add_argument("--dp-clip", type=float, default=1.0,
                        help="Per-update clipping norm C (default: 1.0)")
    parser.add_argument("--dp-delta", type=float, default=1e-5,
                        help="DP delta for the (ε, δ) budget (default: 1e-5)")
    args = parser.parse_args()

    main(args)
