"""Phase 2: Federated Baseline — FedAvg GlobalTCN training (Flower backend).

Uses Flower (flwr) to simulate Federated Averaging (FedAvg) across all
clients on a single machine.  Trains a Global TCN point-forecast model (no
ResidualCorrector) to produce the ``Y_pre`` baseline.  This serves as the
control against which Phase 3 personalization gains are measured.

Usage::

    python -m fl_code.train_baseline                              # defaults
    python -m fl_code.train_baseline --rounds 30 --lr 0.001       # custom
    python -m fl_code.train_baseline --clients steel_ind_0 eld_ind_0  # subset
    python -m fl_code.train_baseline --max-seqs 5 --eval-seqs 3   # fast dev
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
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import DifferentialPrivacyServerSideFixedClipping
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
OUTPUT_DIR = ROOT / "fl_code" / "baseline_outputs"

# Per-actor data cache.  ``client_fn`` runs inside each Ray actor process;
# this module-level dict lets an actor load its own client's data exactly
# once and reuse it across rounds.  The dict is serialized (empty) with the
# ``client_fn`` closure instead of the ~500 MB full ``cache`` — this is the
# main memory optimisation (see analysis: 13 GB → ~2 GB peak).
_ACTOR_DATA_CACHE: dict[str, dict] = {}


# ============================================================================
# PyTorch <-> Flower ndarray conversion
# ============================================================================

def _state_to_ndarrays(state_dict: dict) -> list[np.ndarray]:
    """PyTorch state_dict → list of numpy arrays (Flower wire format)."""
    return [v.cpu().numpy() for v in state_dict.values()]


def _ndarrays_to_state(ndarrays: list[np.ndarray],
                       keys: list[str]) -> OrderedDict[str, torch.Tensor]:
    """Flower ndarrays → PyTorch state_dict (preserving key order)."""
    return OrderedDict((k, torch.from_numpy(v)) for k, v in zip(keys, ndarrays))


def _extract_train_losses(results) -> dict[str, float]:
    """Per-client train_loss from a round's FitRes list ({cid: loss})."""
    losses: dict[str, float] = {}
    for proxy, fit_res in results:
        loss = fit_res.metrics.get("train_loss")
        if loss is not None:
            cid = fit_res.metrics.get("client_id", proxy.cid)
            losses[cid] = float(loss)
    return losses


# ============================================================================
# Flower Client
# ============================================================================

class _SaveParamsFedAvg(fl.server.strategy.FedAvg):
    """FedAvg that saves the aggregated parameters for post-training eval.

    Records the aggregated parameters, a per-round history of them (for
    round checkpoints), and per-client train losses.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aggregated_parameters = None
        self.round_parameters: list = []
        self.per_client_train_losses: dict[int, dict[str, float]] = {}

    def aggregate_fit(self, server_round, results, failures):
        agg = super().aggregate_fit(server_round, results, failures)
        if agg is not None:
            self.aggregated_parameters = agg[0]
            self.round_parameters.append(agg[0])
        self.per_client_train_losses[server_round] = _extract_train_losses(results)
        return agg


class _SaveParamsDPFedAvg(DifferentialPrivacyServerSideFixedClipping):
    """DP-FedAvg (server-side clipping) that also saves the **noised**
    aggregated parameters.

    Important: the saved parameters must be the post-noise ones — saving the
    inner strategy's (un-noised) aggregate would void the (ε, δ) guarantee.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aggregated_parameters = None
        self.round_parameters: list = []
        self.per_client_train_losses: dict[int, dict[str, float]] = {}

    def aggregate_fit(self, server_round, results, failures):
        agg = super().aggregate_fit(server_round, results, failures)
        if agg is not None:
            self.aggregated_parameters = agg[0]
            self.round_parameters.append(agg[0])
        self.per_client_train_losses[server_round] = _extract_train_losses(results)
        return agg


class PowerClient(fl.client.NumPyClient):
    """Per-client wrapper: loads global weights, trains locally, returns update.

    Each client owns its own model + training dataset.  The Flower simulation
    calls ``fit()`` every round with the latest global parameters.
    """

    def __init__(self, cid: str, train_ds: LazySlidingWindowDataset,
                 n_train: int, args: argparse.Namespace):
        self.cid = cid
        self.train_ds = train_ds
        self.n_train = n_train
        self.args = args
        # Detect device inside the Ray actor (may differ from main process)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = build_tcn(TCNConfig()).to(self._device)
        self._state_keys = list(self._model.state_dict().keys())

    def fit(self, parameters, config):
        model = self._model
        state = _ndarrays_to_state(parameters, self._state_keys)
        model.load_state_dict(state)

        loader = DataLoader(
            self.train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)

        for _ in range(self.args.local_epochs):
            loss = train_epoch(model, loader, optimizer, self._device)

        ndarrays = _state_to_ndarrays(model.state_dict())
        return ndarrays, self.n_train, {"train_loss": loss, "client_id": self.cid}

    def evaluate(self, parameters, config):
        # Evaluation is done post-training via evaluate(); skip in-simulation.
        return 0.0, 0, {}


# ============================================================================
# Data helpers
# ============================================================================

def _load_client_cache(client_id: str, stride: int,
                       max_seqs: int | None = None) -> dict:
    """Load + preprocess client data, return dict with train Dataset + metadata."""
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


def _aggregate_fit_metrics(metrics) -> dict:
    """Average per-client train_loss across the round (for the history log)."""
    losses = [m["train_loss"] for _, m in metrics]
    return {"train_loss": float(np.mean(losses)) if losses else float("nan")}


def _compute_epsilon(noise_multiplier: float, rounds: int,
                     sample_rate: float, delta: float) -> float:
    """(ε, δ)-DP budget for DP-FedAvg via RDP composition (Opacus accountant).

    ``sample_rate = num_sampled_clients / num_clients`` — with
    ``fraction_fit=1.0`` every client is sampled each round, so it is 1.0.
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
    init_ndarrays = _state_to_ndarrays(tmp_model.state_dict())

    # Actors load their own data via client_fn — release driver-side
    # training datasets (window lists) to free ~500 MB before simulation.
    for cdata in cache.values():
        cdata.pop("train_ds", None)

    def client_fn(context) -> fl.client.Client:
        """Flower client factory — runs inside each Ray actor.

        Loads the client's data **inside the actor** (cached per actor),
        so the closure stays tiny and the ~500 MB ``cache`` is never
        serialized to actors.
        """
        partition_id = int(context.node_config["partition-id"])
        cid = client_ids[partition_id]
        cdata = _ACTOR_DATA_CACHE.get(cid)
        if cdata is None:
            cdata = _load_client_cache(cid, args.stride, args.max_seqs)
            _ACTOR_DATA_CACHE[cid] = cdata
        return PowerClient(cid, cdata["train_ds"], cdata["n_train"],
                           args).to_client()

    # --- Strategy (custom FedAvg that saves final aggregated parameters) ---
    init_params = ndarrays_to_parameters(init_ndarrays)
    base_strategy = _SaveParamsFedAvg(
        fraction_fit=1.0,               # all clients train each round
        fraction_evaluate=0.0,          # skip built-in eval (we eval post-training)
        min_fit_clients=len(client_ids),
        min_available_clients=len(client_ids),
        initial_parameters=init_params,
        fit_metrics_aggregation_fn=_aggregate_fit_metrics,  # log avg train_loss per round
    )

    # DP-FedAvg (central DP, server-side fixed clipping) when --dp-noise given.
    # Works on the flattened parameter-vector updates, so it is agnostic to
    # the model architecture (TCN included) — unlike local DP-SGD (Opacus).
    dp_info = None
    if args.dp_noise is not None and args.dp_noise > 0:
        strategy = _SaveParamsDPFedAvg(
            strategy=base_strategy,
            noise_multiplier=args.dp_noise,
            clipping_norm=args.dp_clip,
            num_sampled_clients=len(client_ids),
        )
        sample_rate = len(client_ids) / len(client_ids)   # fraction_fit=1.0
        eps = _compute_epsilon(args.dp_noise, args.rounds,
                               sample_rate, args.dp_delta)
        dp_info = {
            "noise_multiplier": args.dp_noise,
            "clipping_norm": args.dp_clip,
            "delta": args.dp_delta,
            "epsilon": round(eps, 4),
        }
        print(f"DP-FedAvg enabled: σ={args.dp_noise}, C={args.dp_clip}, "
              f"(ε={eps:.2f}, δ={args.dp_delta}) after {args.rounds} rounds")
    else:
        strategy = base_strategy

    # --- Save run config (model architecture + hyperparameters) ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tcn_cfg = TCNConfig().to_dict()
    rf = 1 + 2 * (tcn_cfg["kernel_size"] - 1) * (2 ** len(tcn_cfg["num_channels"]) - 1)
    config_json = {
        "script": "fl_code.train_baseline",
        "phase": "Phase 2 — FedAvg GlobalTCN (Flower simulation)",
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
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config_json, f, indent=2, default=str)
    print(f"Saved run config to {OUTPUT_DIR / 'config.json'}")

    # --- Run simulation ---
    print(f"\nStarting Flower simulation: {args.rounds} rounds × {len(client_ids)} clients")

    t0 = time.perf_counter()
    fl_history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_ids),
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )
    elapsed = time.perf_counter() - t0
    print(f"Training complete in {elapsed:.0f}s")

    # --- Extract final aggregated model from strategy ---
    if strategy.aggregated_parameters is None:
        print("WARNING: No aggregated parameters — using initial weights")
        final_ndarrays = init_ndarrays
    else:
        final_ndarrays = parameters_to_ndarrays(strategy.aggregated_parameters)
    state_keys = list(tmp_model.state_dict().keys())
    final_state = _ndarrays_to_state(final_ndarrays, state_keys)
    tmp_model.load_state_dict(final_state)

    # --- Save per-round checkpoints (every aggregated model) ---
    ckpt_dir = OUTPUT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for i, params_obj in enumerate(strategy.round_parameters, start=1):
        ndarrays = parameters_to_ndarrays(params_obj)   # Parameters → ndarray
        state = _ndarrays_to_state(ndarrays, state_keys)
        torch.save(state, ckpt_dir / f"round_{i:03d}.pt")
    print(f"Saved {len(strategy.round_parameters)} round checkpoints to {ckpt_dir}")

    # --- Post-training evaluation ---
    print("\nEvaluating final model on all clients ...")
    eval_device = device
    tmp_model.to(eval_device)
    tmp_model.eval()

    results = _eval_all_clients(tmp_model, cache, args, eval_device)

    print(f"\nFinal — WAPE={results['wape']:.4f}  MAE={results['avg_mae']:.4f}")

    # --- Save outputs (final model = last round checkpoint) ---
    with open(OUTPUT_DIR / "baseline_history.json", "w") as f:
        json.dump({
            "args": {k: str(v) for k, v in vars(args).items()},
            "num_clients": len(client_ids),
            "total_train_windows": total_windows,
            "model_params": n_params,
            "training_time_s": round(elapsed, 1),
            "final_metrics": results,
            "train_losses": [round(loss, 6) for _, loss in
                             fl_history.metrics_distributed_fit.get("train_loss", [])],
            "train_losses_per_client": strategy.per_client_train_losses,
            "final_model": (f"checkpoints/round_{len(strategy.round_parameters):03d}.pt"
                            if strategy.round_parameters else None),
            "dp": dp_info,   # None unless --dp-noise was given
            # Flower's built-in history (fit loss per round, etc.)
            "flower_history": {
                "losses_centralized": fl_history.losses_centralized,
                "losses_distributed": fl_history.losses_distributed,
                "metrics_centralized": fl_history.metrics_centralized,
                "metrics_distributed": fl_history.metrics_distributed,
            },
        }, f, indent=2, default=str)

    # --- Per-client summary ---
    print("\n--- Per-client final metrics ---")
    for cid in client_ids:
        m = results["client_metrics"].get(cid, {})
        print(f"  {cid:20s}  MAE={m.get('mae', float('nan')):.4f}  "
              f"RMSE={m.get('rmse', float('nan')):.4f}  n_train={m.get('n_train', 0)}")

    print(f"\nOutputs saved to {OUTPUT_DIR}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: FedAvg GlobalTCN Baseline Training (Flower backend)",
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

    # DP-FedAvg (central DP, server-side fixed clipping)
    parser.add_argument("--dp-noise", type=float, default=None,
                        help="Enable DP-FedAvg: Gaussian noise multiplier σ "
                             "(default: None = no DP)")
    parser.add_argument("--dp-clip", type=float, default=1.0,
                        help="Per-update clipping norm C (default: 1.0)")
    parser.add_argument("--dp-delta", type=float, default=1e-5,
                        help="DP delta for the (ε, δ) budget (default: 1e-5)")
    args = parser.parse_args()

    main(args)
