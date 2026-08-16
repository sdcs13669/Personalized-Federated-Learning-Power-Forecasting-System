"""Phase 2: Federated Baseline — FedAvg GlobalTCN training.

FedAvg simulation across all clients on one machine via flwr run_simulation
(ray backend; same client/strategy code as the App line).  Trains a Global
TCN point-forecast model (no ResidualCorrector) to produce the ``Y_pre``
baseline.  This serves as the control against which Phase 3 personalization
gains are measured.

DP mode: each client runs **client-side DP-SGD** —
per-sample gradient clipping + Gaussian noise at every local SGD step, so the
uploaded update carries an (ε, δ) guarantee; the server only aggregates and
never sees un-noised gradients.  Two modes:

- ``--dp-epsilon ε``: per-client noise multipliers σᵢ are derived so every
  client spends exactly ε of budget (PLD accounting, default target in docs:
  7.5 ≤ 8).  Big clients get much less noise for the same ε thanks to
  subsampling amplification (q = B/nᵢ).
- ``--dp-noise σ``: uniform σ for all clients (ε printed for reference).

Usage::

    python -m fl_code.train_baseline                              # defaults
    python -m fl_code.train_baseline --rounds 30 --lr 0.001       # custom
    python -m fl_code.train_baseline --clients steel_ind_0 eld_ind_0  # subset
    python -m fl_code.train_baseline --max-seqs 5 --eval-seqs 3   # fast dev
    python -m fl_code.train_baseline --dp-epsilon 7.5 --dp-delta 1e-5  # per-client σᵢ (recommended)
    python -m fl_code.train_baseline --dp-noise 1.0 --dp-delta 1e-5     # uniform σ (ε printed)
    python -m fl_code.train_baseline --output-dir my_run            # custom output root
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from fl_code.train_eval_utils import evaluate
from fl_code.models import TCNConfig, build_tcn
from fl_code.config import (
    INPUT_STEPS, PRED_LEN, STRIDE, TRAIN_RATIO,
    BASELINE_ROUNDS, BASELINE_LOCAL_EPOCHS,
    BASELINE_LR, BASELINE_BATCH_SIZE,
)
from fl_code.fed_core.accounting import (
    dp_epsilon_worst as _dp_epsilon_worst,
    sigma_for_epsilon as _sigma_for_epsilon,
)
from fl_code.fed_core.data import load_client_cache as _load_client_cache

ROOT = Path(__file__).resolve().parents[1]
CLIENT_CONFIG_PATH = ROOT / "fl_code" / "models" / "client_config.yaml"


# ============================================================================
# Data helpers
# ============================================================================

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

    # DP-FedAvg when --dp-epsilon (per-client σᵢ) or --dp-noise (uniform σ)
    # given.  Privacy is enforced **client-side**: each client runs DP-SGD
    # locally — per-sample gradient clipping to l2 ≤ C and Gaussian noise
    # N(0, (σC)²) at every SGD step — before uploading the update; the
    # server (aggregator) never sees un-noised gradients, so the guarantee
    # does not depend on trusting the server.  Per-step mechanism:
    # sensitivity C, noise σC → noise multiplier σ; per-client budget
    # composes all local steps × rounds with sampling rate B/n
    # (subsampling amplification); aggregation is post-processing, so each
    # client's (εᵢ, δ) holds independently and the system-level guarantee
    # is max εᵢ.  In per-client mode σᵢ is derived so every client spends
    # exactly the target ε — big clients (small q = B/nᵢ) need far less
    # noise for the same budget than small ones.
    dp_info = None
    if args.dp_noise is not None or args.dp_epsilon is not None:
        dp_base = {
            "clipping_norm": args.dp_clip,
            "delta": args.dp_delta,
            "accountant": "PLD (dp-accounting, Poisson-subsampled Gaussian)",
        }
        if args.dp_noise is not None:
            client_sizes = [cache[cid]["n_train"] for cid in client_ids]
            eps = _dp_epsilon_worst(client_sizes, args.batch_size,
                                    args.local_epochs, args.rounds,
                                    args.dp_noise, args.dp_delta)
            dp_info = {**dp_base, "mode": "uniform",
                       "noise_multiplier": args.dp_noise,
                       "epsilon": round(eps, 4)}
            print(f"DP-FedAvg enabled (client-side DP-SGD, uniform σ, PLD "
                  f"accounting): per-sample gradient clipping C={args.dp_clip} "
                  f"+ Gaussian noise σ={args.dp_noise} at every local SGD "
                  f"step; (ε={eps:.2f}, δ={args.dp_delta}) after "
                  f"{args.rounds} rounds")
        else:
            per_client: dict[str, dict[str, float]] = {}
            print(f"Deriving per-client sigma for target eps={args.dp_epsilon} "
                  f"(delta={args.dp_delta}, PLD accounting) ...")
            for cid in client_ids:
                sigma_i, eps_i = _sigma_for_epsilon(
                    cache[cid]["n_train"], args.batch_size,
                    args.local_epochs, args.rounds, args.dp_delta,
                    args.dp_epsilon)
                per_client[cid] = {"sigma": round(sigma_i, 4),
                                   "epsilon": round(eps_i, 4)}
                print(f"  {cid:<20s} n={cache[cid]['n_train']:>10,}  "
                      f"sigma={sigma_i:.3f}  eps={eps_i:.3f}")
            system_eps = max(p["epsilon"] for p in per_client.values())
            dp_info = {**dp_base, "mode": "per-client",
                       "target_epsilon": args.dp_epsilon,
                       "epsilon": round(system_eps, 4),
                       "per_client": per_client}
            print(f"DP-FedAvg enabled (client-side DP-SGD, per-client sigma): "
                  f"clipping C={args.dp_clip}, system eps = max eps_i = "
                  f"{system_eps:.3f} <= 8 OK")

    # DP and non-DP runs write to separate sub-directories so they never
    # overwrite each other.
    variant_dir = Path(args.output_dir) / ("dp" if dp_info else "nodp")
    variant_dir.mkdir(parents=True, exist_ok=True)

    # --- Save run config (model architecture + hyperparameters) ---
    tcn_cfg = TCNConfig().to_dict()
    rf = 1 + 2 * (tcn_cfg["kernel_size"] - 1) * (2 ** len(tcn_cfg["num_channels"]) - 1)
    config_json = {
        "script": "fl_code.train_baseline",
        "phase": "Phase 2 — FedAvg GlobalTCN (flwr simulation)",
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

    # --- Run FedAvg rounds via flwr simulation (ray backend) ---
    print(f"\nStarting FedAvg simulation (flwr/ray): {args.rounds} rounds × "
          f"{len(client_ids)} clients")

    ckpt_dir = variant_dir / "checkpoints"
    state_keys = list(tmp_model.state_dict().keys())
    audit_path = variant_dir / "audit_log.json"
    task_cfg = {
        "lr": args.lr, "batch_size": args.batch_size,
        "local_epochs": args.local_epochs,
        "dp_mode": ("none" if dp_info is None
                    else ("uniform" if args.dp_noise is not None
                          else "per_client")),
        "dp_clip": args.dp_clip, "dp_delta": args.dp_delta,
        "dp_sigma": (float(args.dp_noise) if args.dp_noise is not None
                     else None),
        "dp_target_epsilon": (float(args.dp_epsilon)
                              if args.dp_epsilon is not None else None),
    }
    task = {
        "name": f"baseline_{'dp' if dp_info else 'nodp'}",
        "rounds": args.rounds, "round_timeout": None,
        "checkpoint_dir": str(ckpt_dir),
        "audit_path": str(audit_path),
        "expected_clients": client_ids, "deliver_model": False,
        "started_at": None, "cfg": task_cfg,
    }
    from fl_code.fed_core.server_core import build_server_app, make_client_fn
    from flwr.client import ClientApp
    from flwr.simulation import run_simulation

    t0 = time.perf_counter()
    run_simulation(
        server_app=build_server_app(task, state_keys),
        client_app=ClientApp(client_fn=make_client_fn(
            cache, client_ids, state_keys,
            {"lr": args.lr, "batch_size": args.batch_size,
             "local_epochs": args.local_epochs, "device": device})),
        num_supernodes=len(client_ids),
        backend_config={
            "client_resources": (
                {"num_cpus": 2, "num_gpus": 0.5}
                if device == "cuda"
                else {"num_cpus": 2, "num_gpus": 0})
        },
    )
    elapsed = time.perf_counter() - t0

    with open(audit_path) as f:
        audit = json.load(f)
    per_client_losses = {row["round"]: row["client_losses"]
                         for row in audit["rounds"]}
    print(f"Training complete in {elapsed:.0f}s")
    print(f"Saved {args.rounds} round checkpoints to {ckpt_dir}")

    # --- Load final aggregated model for eval ---
    tmp_model.load_state_dict(torch.load(
        ckpt_dir / f"round_{args.rounds:03d}.pt", weights_only=True))

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
                    "(flwr simulation, ray backend)",
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

    # DP-FedAvg — client-side DP-SGD (per-sample gradient clip + noise)
    parser.add_argument("--dp-noise", type=float, default=None,
                        help="Enable DP-FedAvg via client-side DP-SGD with "
                             "a uniform noise multiplier σ for all clients "
                             "(ε printed for reference); mutually exclusive "
                             "with --dp-epsilon (default: None = no DP)")
    parser.add_argument("--dp-epsilon", type=float, default=None,
                        help="Enable DP-FedAvg via client-side DP-SGD with "
                             "per-client noise multipliers derived so every "
                             "client spends exactly eps of budget (PLD "
                             "accounting; e.g. 7.5, must be <= 8 per master "
                             "plan); mutually exclusive with --dp-noise "
                             "(default: None = no DP)")
    parser.add_argument("--dp-clip", type=float, default=1.0,
                        help="Per-sample gradient clipping norm C "
                             "(default: 1.0)")
    parser.add_argument("--dp-delta", type=float, default=1e-5,
                        help="DP delta for the (ε, δ) budget (default: 1e-5)")
    args = parser.parse_args()

    if args.dp_noise is not None and args.dp_epsilon is not None:
        parser.error("--dp-noise and --dp-epsilon are mutually exclusive")

    main(args)
