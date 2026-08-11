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
from torch.utils.data import DataLoader

from fl_code.data_utils import (
    load_client_data,
    preprocess,
    LazySlidingWindowDataset,
)
from fl_code.train_eval_utils import train_epoch, evaluate
from fl_code.models import TCNConfig, build_tcn

ROOT = Path(__file__).resolve().parents[1]
CLIENT_CONFIG_PATH = ROOT / "fl_code" / "models" / "client_config.yaml"
OUTPUT_DIR = ROOT / "fl_code" / "outputs"


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


# ============================================================================
# Flower Client
# ============================================================================

class _SaveParamsFedAvg(fl.server.strategy.FedAvg):
    """FedAvg that saves the final aggregated parameters for post-training eval."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aggregated_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        agg = super().aggregate_fit(server_round, results, failures)
        if agg is not None:
            self.aggregated_parameters = agg[0]
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
        return ndarrays, self.n_train, {"train_loss": loss}

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

    def client_fn(context) -> fl.client.Client:
        """Flower client factory — called once per client at simulation start."""
        partition_id = int(context.node_config["partition-id"])
        cid = client_ids[partition_id]
        cdata = cache[cid]
        return PowerClient(cid, cdata["train_ds"], cdata["n_train"], args).to_client()

    # --- Strategy (custom FedAvg that saves final aggregated parameters) ---
    init_params = ndarrays_to_parameters(init_ndarrays)
    strategy = _SaveParamsFedAvg(
        fraction_fit=1.0,               # all clients train each round
        fraction_evaluate=0.0,          # skip built-in eval (we eval post-training)
        min_fit_clients=len(client_ids),
        min_available_clients=len(client_ids),
        initial_parameters=init_params,
    )

    # --- Run simulation ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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

    # --- Post-training evaluation ---
    print("\nEvaluating final model on all clients ...")
    eval_device = device
    tmp_model.to(eval_device)
    tmp_model.eval()

    results = _eval_all_clients(tmp_model, cache, args, eval_device)

    print(f"\nFinal — WAPE={results['wape']:.4f}  MAE={results['avg_mae']:.4f}")

    # --- Save outputs ---
    torch.save(tmp_model.state_dict(), OUTPUT_DIR / "best_global_tcn.pt")

    with open(OUTPUT_DIR / "baseline_history.json", "w") as f:
        json.dump({
            "args": {k: str(v) for k, v in vars(args).items()},
            "num_clients": len(client_ids),
            "total_train_windows": total_windows,
            "model_params": n_params,
            "training_time_s": round(elapsed, 1),
            "final_metrics": results,
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
    parser.add_argument("--rounds", type=int, default=20,
                        help="Communication rounds (default: 20)")
    parser.add_argument("--local-epochs", type=int, default=1,
                        help="Local SGD epochs per round (default: 1)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--stride", type=int, default=48,
                        help="Sliding-window stride (default: 48)")
    parser.add_argument("--eval-seqs", type=int, default=None,
                        help="Cap eval to first N sequences per client (default: all)")
    parser.add_argument("--max-seqs", type=int, default=None,
                        help="Cap training sequences per client (default: all)")
    parser.add_argument("--clients", nargs="*", default=None,
                        help="Client ids to include (default: all)")
    args = parser.parse_args()

    main(args)
