"""Phase 4: DP-Personalised FL — Global TCN + per-client DP Residual Corrector.

Loads a frozen Global TCN (Phase 2), then trains a local Residual Corrector
for each client with **DP-SGD** (Opacus) at multiple noise levels.  Compares
DP vs non-DP performance and produces ε-WAPE data for privacy-utility
trade-off analysis.

Usage::

    python -m fl_code.train_dp_personalized
    python -m fl_code.train_dp_personalized --clients steel_ind_0
    python -m fl_code.train_dp_personalized --noise-multipliers 0.5 1.0 2.0 5.0
    python -m fl_code.train_dp_personalized --max-seqs 5 --epochs 10
    python -m fl_code.train_dp_personalized --rc-type mlp     # safest for DP
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from fl_code.data_utils import (
    load_client_data,
    preprocess,
    LazySlidingWindowDataset,
)
from fl_code.models import (
    TCNConfig, CorrectorConfig,
    build_tcn, build_corrector,
)
from fl_code.models.rc import quantile_loss
from fl_code.config import (
    STRIDE, CORRECTOR_EPOCHS, CORRECTOR_LR, CORRECTOR_BATCH_SIZE,
    DP_NOISE_MULTIPLIERS, DP_MAX_GRAD_NORM, DP_DELTA,
)

# Import shared Phase 3 components
from fl_code.train_personalized import (
    _LazyWindowsWithLocal,
    _precompute_ypre,
    CorrectorDataset,
    _evaluate_personalized,
    _load_client_data_cached,
    _list_clients,
)

ROOT = Path(__file__).resolve().parents[1]
CLIENT_CONFIG_PATH = ROOT / "fl_code" / "models" / "client_config.yaml"
OUTPUT_DIR = ROOT / "fl_code" / "dp_outputs"


# ---------------------------------------------------------------------------
# DP training with Opacus
# ---------------------------------------------------------------------------

def _train_corrector_dp(
    train_ds,
    corr_cfg: CorrectorConfig,
    device: str,
    epochs: int,
    lr: float,
    batch_size: int,
    max_grad_norm: float,
    noise_multiplier: float,
    delta: float,
) -> tuple[nn.Module, float, list[float]]:
    """Train a Corrector with DP-SGD via Opacus PrivacyEngine.

    Returns ``(trained_corrector, epsilon, epoch_losses)``.
    If Opacus fails (e.g. unsupported layer), raises RuntimeError with
    guidance to switch ``--rc-type mlp``.
    """
    from opacus import PrivacyEngine

    corrector = build_corrector(corr_cfg).to(device)
    quantiles = corrector.quantiles  # save ref before GradSampleModule wrapping

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        drop_last=False)
    optimizer = torch.optim.Adam(corrector.parameters(), lr=lr)

    try:
        privacy_engine = PrivacyEngine()
        corrector, optimizer, loader = privacy_engine.make_private(
            module=corrector,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            poisson_sampling=True,
            loss_reduction="mean",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Opacus PrivacyEngine failed for rc_type={corr_cfg.rc_type!r}. "
            f"Try --rc-type mlp (the safest option for DP). "
            f"Original error: {exc}"
        ) from exc

    epoch_losses = []
    for epoch in range(epochs):
        corrector.train()
        total_loss = 0.0
        n = 0

        for y_pre, residual, x_local, target in loader:
            y_pre = y_pre.to(device)
            residual = residual.to(device)
            target = target.to(device)
            if x_local.numel() > 0:
                x_local = x_local.to(device)
            else:
                x_local = None

            optimizer.zero_grad()
            out = corrector(y_pre, residual, x_local)
            loss = quantile_loss(target, out, quantiles)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n += 1

        avg_loss = total_loss / max(n, 1)
        epoch_losses.append(avg_loss)

    try:
        epsilon = float(privacy_engine.get_epsilon(delta))
    except Exception:
        epsilon = float("nan")

    return corrector, epsilon, epoch_losses


# ---------------------------------------------------------------------------
# Non-DP training (baseline reference)
# ---------------------------------------------------------------------------

def _train_corrector_baseline(
    train_ds,
    corr_cfg: CorrectorConfig,
    device: str,
    epochs: int,
    lr: float,
    batch_size: int,
) -> tuple[nn.Module, list[float]]:
    """Train a Corrector **without** DP. Returns ``(corrector, epoch_losses)``."""
    corrector = build_corrector(corr_cfg).to(device)
    quantiles = corrector.quantiles

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        drop_last=False)
    optimizer = torch.optim.Adam(corrector.parameters(), lr=lr)

    epoch_losses = []
    for _ in range(epochs):
        corrector.train()
        total_loss = 0.0
        n = 0

        for y_pre, residual, x_local, target in loader:
            y_pre = y_pre.to(device)
            residual = residual.to(device)
            target = target.to(device)
            if x_local.numel() > 0:
                x_local = x_local.to(device)
            else:
                x_local = None

            optimizer.zero_grad()
            out = corrector(y_pre, residual, x_local)
            loss = quantile_loss(target, out, quantiles)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n += 1

        epoch_losses.append(total_loss / max(n, 1))

    return corrector, epoch_losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"DP: max_grad_norm={args.max_grad_norm}, delta={args.delta}")
    print(f"Noise multipliers: {args.noise_multipliers}")
    print(f"Corrector type: {args.rc_type}")

    # ---- Load frozen Global TCN ----
    global_path = Path(args.global_model)
    if not global_path.exists():
        raise FileNotFoundError(
            f"Global model not found: {global_path}.  "
            f"Run Phase 2 first (train_baseline.py)."
        )
    print(f"Loading Global TCN: {global_path}")
    global_tcn = build_tcn(TCNConfig()).to(device)
    global_tcn.load_state_dict(torch.load(global_path, map_location=device,
                                          weights_only=True))
    global_tcn.eval()
    for p in global_tcn.parameters():
        p.requires_grad = False

    # ---- Clients ----
    client_ids = _list_clients(args.clients)
    print(f"Clients ({len(client_ids)}): {', '.join(client_ids)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = {}

    for cid in client_ids:
        print(f"\n{'='*60}")
        print(f"Client: {cid}")
        print(f"{'='*60}")

        data = _load_client_data_cached(cid, args.stride, args.max_seqs)
        n_seqs = len(data["seqs"])
        local_dim = len(data["local_cols"])

        # ---- Pre-compute Y_pre (shared across all noise levels) ----
        t0 = time.perf_counter()
        print(f"  Pre-computing Y_pre ({n_seqs} sequences, local_dim={local_dim})...")
        y_pre, y_true, x_local = _precompute_ypre(
            global_tcn, data["df_norm"], data["seqs"],
            data["public_cols"], data["local_cols"],
            args.stride, device,
        )
        n_windows = len(y_pre)
        print(f"  {n_windows} windows ({time.perf_counter() - t0:.1f}s)")

        if n_windows < 10:
            print(f"  WARNING: too few windows, skipping")
            continue

        train_ds = CorrectorDataset(y_pre, y_true, x_local)
        corr_cfg = CorrectorConfig(
            rc_type=args.rc_type,
            local_feat_dim=local_dim,
        )

        eval_seqs = data["seqs"]
        if args.eval_seqs and len(eval_seqs) > args.eval_seqs:
            eval_seqs = eval_seqs[:args.eval_seqs]

        client_result = {
            "n_windows": n_windows,
            "n_seqs": n_seqs,
            "local_dim": local_dim,
            "corrector_type": args.rc_type,
            "delta": args.delta,
            "noise_multipliers": args.noise_multipliers,
            "baseline": None,
            "dp_runs": [],
        }

        # ---- Non-DP baseline ----
        print(f"\n  --- Non-DP Baseline ---")
        t0 = time.perf_counter()
        corr_base, losses_base = _train_corrector_baseline(
            train_ds, corr_cfg, device,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        )
        train_time = time.perf_counter() - t0

        mae_base, mae_nodp = _evaluate_personalized(
            global_tcn, corr_base, data["df_norm"],
            eval_seqs, data["public_cols"], data["local_cols"],
            args.stride, device,
        )
        gain = (mae_base - mae_nodp) / mae_base * 100 if mae_base == mae_base else float("nan")

        client_result["baseline"] = {
            "mae_ypre": mae_base,
            "mae_personalized": mae_nodp,
            "improvement_pct": round(gain, 2),
            "final_loss": round(losses_base[-1], 6),
            "train_time_s": round(train_time, 1),
        }
        print(f"  Baseline MAE: {mae_base:.4f} → {mae_nodp:.4f}  ({gain:+.1f}%)")

        # Save non-DP corrector
        torch.save(corr_base.state_dict(), OUTPUT_DIR / f"corrector_{cid}_nodp.pt")
        del corr_base  # free GPU
        torch.cuda.empty_cache() if device == "cuda" else None

        # ---- DP training at multiple noise levels ----
        for noise in args.noise_multipliers:
            print(f"\n  --- DP: σ={noise} ---")
            t0 = time.perf_counter()

            corr_dp, epsilon, losses_dp = _train_corrector_dp(
                train_ds, corr_cfg, device,
                epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                max_grad_norm=args.max_grad_norm,
                noise_multiplier=noise,
                delta=args.delta,
            )
            train_time = time.perf_counter() - t0

            _, mae_dp = _evaluate_personalized(
                global_tcn, corr_dp, data["df_norm"],
                eval_seqs, data["public_cols"], data["local_cols"],
                args.stride, device,
            )

            dp_entry = {
                "noise_multiplier": noise,
                "epsilon": round(epsilon, 4) if not np.isnan(epsilon) else None,
                "mae_personalized": mae_dp,
                "final_loss": round(losses_dp[-1], 6),
                "train_time_s": round(train_time, 1),
            }
            client_result["dp_runs"].append(dp_entry)

            eps_s = f"ε={epsilon:.2f}" if not np.isnan(epsilon) else "ε=N/A"
            print(f"  DP MAE: {mae_dp:.4f}  ({eps_s})  vs non-DP: {mae_nodp:.4f}")

            # Save DP corrector
            suffix = f"sigma{str(noise).replace('.', '_')}"
            torch.save(corr_dp.state_dict(), OUTPUT_DIR / f"corrector_{cid}_dp_{suffix}.pt")
            del corr_dp
            torch.cuda.empty_cache() if device == "cuda" else None

        all_results[cid] = client_result

    # ---- Summary table ----
    print(f"\n{'='*75}")
    print("Summary — Global TCN vs Non-DP vs DP (P50 MAE)")
    print(f"{'='*75}")

    for cid, r in all_results.items():
        bl = r["baseline"]
        print(f"\n  [{cid}]")
        print(f"    Global TCN (Y_pre):     {bl['mae_ypre']:.4f}")
        print(f"    Non-DP (ε=∞):            {bl['mae_personalized']:.4f}  "
              f"({bl['improvement_pct']:+.1f}%)")
        for entry in r["dp_runs"]:
            eps_s = f"ε={entry['epsilon']:.2f}" if entry["epsilon"] else "ε=N/A"
            sigma_s = f"σ={entry['noise_multiplier']:.1f}"
            print(f"    DP ({sigma_s}, {eps_s}):  {entry['mae_personalized']:.4f}")

    # ---- Export JSON ----
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = OUTPUT_DIR / f"dp_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump({
            "global_model": str(global_path),
            "args": {k: str(v) for k, v in vars(args).items()},
            "results": all_results,
        }, f, indent=2, default=str)

    print(f"\nResults → {results_path}")
    print(f"Outputs → {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 4: DP-Personalised FL — Global TCN + DP Residual Corrector",
    )
    parser.add_argument(
        "--global-model", type=str,
        default=str(ROOT / "fl_code" / "baseline_outputs" / "best_global_tcn.pt"),
        help="Path to frozen Global TCN checkpoint",
    )
    parser.add_argument(
        "--epochs", type=int, default=CORRECTOR_EPOCHS,
        help=f"Corrector training epochs per run (default: {CORRECTOR_EPOCHS})",
    )
    parser.add_argument(
        "--lr", type=float, default=CORRECTOR_LR,
        help=f"Learning rate (default: {CORRECTOR_LR})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=CORRECTOR_BATCH_SIZE,
        help=f"Batch size (default: {CORRECTOR_BATCH_SIZE})",
    )
    parser.add_argument(
        "--stride", type=int, default=STRIDE,
        help=f"Sliding-window stride (default: {STRIDE}, "
             f"= pred_len for continuous coverage)",
    )
    parser.add_argument(
        "--eval-seqs", type=int, default=None,
        help="Cap eval to first N sequences per client",
    )
    parser.add_argument(
        "--max-seqs", type=int, default=None,
        help="Cap training sequences per client",
    )
    parser.add_argument(
        "--clients", nargs="*", default=None,
        help="Client ids to include (default: all)",
    )

    # DP-specific
    parser.add_argument(
        "--noise-multipliers", type=float, nargs="+",
        default=list(DP_NOISE_MULTIPLIERS),
        help=f"Noise multipliers σ for DP-SGD (default: {' '.join(map(str, DP_NOISE_MULTIPLIERS))})",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=DP_MAX_GRAD_NORM,
        help=f"Per-sample gradient clipping norm C (default: {DP_MAX_GRAD_NORM})",
    )
    parser.add_argument(
        "--delta", type=float, default=DP_DELTA,
        help=f"DP delta parameter — should be < 1/N (default: {DP_DELTA})",
    )
    parser.add_argument(
        "--rc-type", type=str, default="mlp",
        choices=["mlp", "lstm", "tcn"],
        help="Corrector architecture (default: mlp — safest DP compat)",
    )

    args = parser.parse_args()
    main(args)
