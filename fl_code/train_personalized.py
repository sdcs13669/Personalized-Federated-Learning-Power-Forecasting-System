"""Phase 3: Personalised FL — Global TCN + per-client Residual Corrector.

Loads a frozen Global TCN (Phase 2), then trains a local Residual Corrector
for each client with pinball loss.  The Corrector predicts 3-quantile residual
corrections ``E_corr``, producing ``Y_final = Y_pre + E_corr``.

Usage::

    python -m fl_code.train_personalized
    python -m fl_code.train_personalized --global-model outputs/best_global_tcn.pt
    python -m fl_code.train_personalized --clients steel_ind_0 tetouan_city_0
    python -m fl_code.train_personalized --max-seqs 5 --epochs 20
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
from torch.utils.data import DataLoader, Dataset

from fl_code.data_utils import (
    load_client_data,
    preprocess,
    LazySlidingWindowDataset,
)
from fl_code.train_eval_utils import evaluate
from fl_code.models import (
    TCNConfig, CorrectorConfig,
    build_tcn, build_corrector,
)
from fl_code.models.rc import quantile_loss
from fl_code.config import (
    STRIDE, CORRECTOR_EPOCHS, CORRECTOR_LR, CORRECTOR_BATCH_SIZE,
)

ROOT = Path(__file__).resolve().parents[1]
CLIENT_CONFIG_PATH = ROOT / "fl_code" / "models" / "client_config.yaml"
OUTPUT_DIR = ROOT / "fl_code" / "personalized_outputs"

# ---------------------------------------------------------------------------
# Lazy dataset variant with local features
# ---------------------------------------------------------------------------


class _LazyWindowsWithLocal(LazySlidingWindowDataset):
    """LazySlidingWindowDataset that also returns local features on __getitem__."""

    def __init__(self, df, seqs, public_cols, local_cols=None,
                 input_steps=144, pred_len=6, stride=48,
                 train=True, train_ratio=0.8):
        super().__init__(df, seqs, public_cols, input_steps=input_steps,
                         pred_len=pred_len, stride=stride, train=train,
                         train_ratio=train_ratio)
        self.local_cols = local_cols or []
        self.loc_arr = df[self.local_cols].values.astype(np.float32) if self.local_cols else None

    def __getitem__(self, idx):
        X, y = super().__getitem__(idx)
        _, start = self.windows[idx]
        in_end = start + self.input_steps
        out_end = in_end + self.pred_len

        if self.loc_arr is not None:
            x_local = torch.from_numpy(self.loc_arr[in_end:out_end].copy())
        else:
            x_local = torch.empty(0)

        return X, y, x_local


# ---------------------------------------------------------------------------
# Pre-compute Y_pre from frozen Global TCN
# ---------------------------------------------------------------------------

@torch.no_grad()
def _precompute_ypre(model, df_norm, seqs, public_cols, local_cols,
                     stride, device) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Stream through all training windows, collect (Y_pre, y_true, X_local)."""
    ds = _LazyWindowsWithLocal(
        df_norm, seqs, public_cols, local_cols,
        stride=stride, train=True,
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False, drop_last=False)

    ypre_list, y_list, loc_list = [], [], []
    for batch in loader:
        X = batch[0].to(device)
        ypre_list.append(model(X).cpu().numpy())
        y_list.append(batch[1].cpu().numpy())
        if ds.loc_arr is not None:
            loc_list.append(batch[2].cpu().numpy())

    y_pre = np.concatenate(ypre_list, axis=0)
    y_true = np.concatenate(y_list, axis=0)
    x_local = np.concatenate(loc_list, axis=0) if loc_list else None
    return y_pre, y_true, x_local


# ---------------------------------------------------------------------------
# Corrector training dataset
# ---------------------------------------------------------------------------

class CorrectorDataset(Dataset):
    """Produces (y_pre, residual_history, x_local_dynamic, target) tuples.

    residual_history[i] = y_true[i-1] - y_pre[i-1]  (previous-window error)
    target[i]           = y_true[i]   - y_pre[i]    (current-window error)

    Index 0 is skipped (no previous residual), so len = N - 1.
    """

    def __init__(self, y_pre: np.ndarray, y_true: np.ndarray,
                 x_local: np.ndarray | None = None):
        self.y_pre = torch.from_numpy(y_pre)
        self.y_true = torch.from_numpy(y_true)
        self.x_local = torch.from_numpy(x_local) if x_local is not None else None

    def __len__(self):
        return len(self.y_pre) - 1

    def __getitem__(self, idx):
        i = idx + 1  # offset: need previous window for residual_history
        residual = self.y_true[i - 1] - self.y_pre[i - 1]
        target = self.y_true[i] - self.y_pre[i]
        x_loc = self.x_local[i] if self.x_local is not None else torch.empty(0)

        return self.y_pre[i], residual, x_loc, target


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_corrector_epoch(corrector, loader, optimizer, device):
    """One epoch of Corrector training with pinball loss."""
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
        loss = quantile_loss(target, out, corrector.quantiles)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Rolling-forecast evaluation with Corrector
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate_personalized(global_model, corrector, df_norm, seqs,
                           public_cols, local_cols, stride, device):
    """Rolling-forecast eval: Global TCN + Corrector → Y_final.

    Returns (baseline_mae, personal_mae) for the Global TCN P50 vs Corrector P50.
    """
    pub_arr = df_norm[public_cols].values.astype(np.float32)
    loc_arr = df_norm[local_cols].values.astype(np.float32) if local_cols else None
    input_steps = 144
    pred_len = 6

    all_preds_base = []
    all_preds_corr = []
    all_actuals = []

    for s in seqs:
        load = df_norm[s].values.astype(np.float32)
        f = df_norm[s].first_valid_index()
        l = df_norm[s].last_valid_index()
        if f is None or l is None:
            continue

        valid_len = l - f + 1
        split = f + int(valid_len * 0.8)

        prev_residual = np.zeros(pred_len, dtype=np.float32)  # initial residual=0
        pos = split

        while pos + input_steps + pred_len <= l + 1:
            # Global TCN prediction
            X_pub = pub_arr[pos:pos + input_steps].T
            X_load = load[pos:pos + input_steps][np.newaxis, :]
            X = np.concatenate([X_pub, X_load], axis=0)
            X_t = torch.from_numpy(X).unsqueeze(0).to(device)
            y_pre = global_model(X_t).squeeze(0)  # (pred_len,)

            # Corrector prediction
            y_pre_np = y_pre.cpu().numpy()
            residual_t = torch.from_numpy(prev_residual).unsqueeze(0).to(device)
            if loc_arr is not None:
                x_loc = loc_arr[pos + input_steps:pos + input_steps + pred_len]
                x_loc_t = torch.from_numpy(x_loc).unsqueeze(0).to(device)
            else:
                x_loc_t = None
            e_corr = corrector(y_pre.unsqueeze(0), residual_t, x_loc_t).squeeze(0)
            y_final = y_pre_np[:, np.newaxis] + e_corr.cpu().numpy()  # (pred_len, 3)

            actual = load[pos + input_steps:pos + input_steps + pred_len]

            all_preds_base.append(y_pre_np)
            all_preds_corr.append(y_final)
            all_actuals.append(actual)

            # Update residual for next step
            prev_residual = actual - y_pre_np
            pos += stride

    if not all_actuals:
        return float("nan"), float("nan")

    actuals = np.concatenate(all_actuals)
    valid = ~np.isnan(actuals)

    base_preds = np.concatenate(all_preds_base)[valid]
    corr_preds = np.concatenate(all_preds_corr)[valid]  # (T, 3)
    actuals = actuals[valid]

    mae_base = float(np.mean(np.abs(base_preds - actuals)))
    mae_corr = float(np.mean(np.abs(corr_preds[:, 1] - actuals)))  # P50 = index 1

    return mae_base, mae_corr


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_client_data_cached(client_id: str, stride: int,
                             max_seqs: int | None = None) -> dict:
    """Load + preprocess, return normalised df and metadata."""
    df, info = load_client_data(client_id)
    feat_names = set(info["public_features"] + info["local_features"])
    seqs = [c for c in df.columns if c not in feat_names and c != "datetime"]

    if max_seqs and len(seqs) > max_seqs:
        seqs = seqs[:max_seqs]

    df_norm, _ = preprocess(df, seqs, info["local_features"])
    return {
        "df_norm": df_norm,
        "seqs": seqs,
        "public_cols": info["public_features"],
        "local_cols": info["local_features"],
    }


def _list_clients(whitelist: list[str] | None = None) -> list[str]:
    with open(CLIENT_CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    ids: list[str] = []
    for ds_cfg in config.values():
        for cid in ds_cfg["clients"]:
            ids.append(cid)
    if whitelist:
        ids = [c for c in ids if c in whitelist]
    return ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load frozen Global TCN ---
    global_path = Path(args.global_model)
    if not global_path.exists():
        raise FileNotFoundError(f"Global model not found: {global_path}.  "
                                f"Run Phase 2 first (train_baseline.py).")
    print(f"Loading Global TCN: {global_path}")
    global_tcn = build_tcn(TCNConfig()).to(device)
    global_tcn.load_state_dict(torch.load(global_path, map_location=device,
                                          weights_only=True))
    global_tcn.eval()
    for p in global_tcn.parameters():
        p.requires_grad = False

    # --- Clients ---
    client_ids = _list_clients(args.clients)
    print(f"Clients ({len(client_ids)}): {', '.join(client_ids)}")

    # --- Per-client training ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = {}

    for cid in client_ids:
        print(f"\n{'='*60}")
        print(f"Client: {cid}")
        print(f"{'='*60}")

        data = _load_client_data_cached(cid, args.stride, args.max_seqs)
        n_seqs = len(data["seqs"])
        local_dim = len(data["local_cols"])

        # ---- Pre-compute Y_pre ----
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

        # ---- Build Corrector ----
        corr_cfg = CorrectorConfig(local_feat_dim=local_dim)
        corrector = build_corrector(corr_cfg).to(device)
        print(f"  Corrector: {corr_cfg.rc_type}, {corr_cfg.local_feat_dim} local dims, "
              f"{sum(p.numel() for p in corrector.parameters()):,} params")

        # ---- Train Corrector ----
        train_ds = CorrectorDataset(y_pre, y_true, x_local)
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            drop_last=False)
        optimizer = torch.optim.Adam(corrector.parameters(), lr=args.lr)

        t0 = time.perf_counter()
        for epoch in range(args.epochs):
            loss = _train_corrector_epoch(corrector, loader, optimizer, device)
            if (epoch + 1) % max(1, args.epochs // 5) == 0:
                print(f"  Epoch {epoch + 1:3d}/{args.epochs}  loss={loss:.6f}")
        print(f"  Training: {time.perf_counter() - t0:.1f}s")

        # ---- Evaluate ----
        print(f"  Evaluating rolling forecast ...")
        t0 = time.perf_counter()
        eval_seqs = data["seqs"]
        if args.eval_seqs and len(eval_seqs) > args.eval_seqs:
            eval_seqs = eval_seqs[:args.eval_seqs]

        mae_base, mae_corr = _evaluate_personalized(
            global_tcn, corrector, data["df_norm"],
            eval_seqs, data["public_cols"], data["local_cols"],
            args.stride, device,
        )
        print(f"  Eval: {time.perf_counter() - t0:.1f}s")
        print(f"  Baseline MAE (Y_pre):  {mae_base:.4f}")
        print(f"  Personal MAE (Y_final): {mae_corr:.4f}")
        if not np.isnan(mae_base) and not np.isnan(mae_corr):
            gain = (mae_base - mae_corr) / mae_base * 100
            print(f"  Improvement: {gain:+.1f}%")

        # ---- Save Corrector ----
        torch.save(corrector.state_dict(),
                   OUTPUT_DIR / f"corrector_{cid}.pt")

        all_results[cid] = {
            "mae_baseline": mae_base,
            "mae_personalized": mae_corr,
            "improvement_pct": round(gain, 2) if not np.isnan(mae_base) else None,
            "n_windows": n_windows,
            "n_seqs": n_seqs,
            "local_dim": local_dim,
            "corrector_type": corr_cfg.rc_type,
        }

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Summary — Global TCN vs Personalised (P50)")
    print(f"{'='*60}")
    print(f"{'Client':20s}  {'Y_pre MAE':>10s}  {'Y_final MAE':>10s}  {'Delta':>8s}")
    print("-" * 52)
    for cid, r in all_results.items():
        base = r["mae_baseline"]
        corr = r["mae_personalized"]
        delta = f"{(base - corr) / base * 100:+.1f}%" if base == base else "N/A"
        print(f"  {cid:20s}  {base:10.4f}  {corr:10.4f}  {delta:>8s}")

    # --- Save results ---
    with open(OUTPUT_DIR / "personalized_results.json", "w") as f:
        json.dump({
            "global_model": str(global_path),
            "args": {k: str(v) for k, v in vars(args).items()},
            "results": all_results,
        }, f, indent=2, default=str)

    print(f"\nOutputs saved to {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3: Personalised FL — Global TCN + Residual Corrector",
    )
    parser.add_argument("--global-model", type=str,
                        default=str(ROOT / "fl_code" / "baseline_outputs" / "best_global_tcn.pt"),
                        help="Path to frozen Global TCN checkpoint")
    parser.add_argument("--epochs", type=int, default=CORRECTOR_EPOCHS,
                        help=f"Corrector training epochs per client (default: {CORRECTOR_EPOCHS})")
    parser.add_argument("--lr", type=float, default=CORRECTOR_LR,
                        help=f"Learning rate (default: {CORRECTOR_LR})")
    parser.add_argument("--batch-size", type=int, default=CORRECTOR_BATCH_SIZE,
                        help=f"Batch size for Corrector training (default: {CORRECTOR_BATCH_SIZE})")
    parser.add_argument("--stride", type=int, default=STRIDE,
                        help=f"Sliding-window stride (default: {STRIDE}, "
                             f"= pred_len for continuous coverage)")
    parser.add_argument("--eval-seqs", type=int, default=None,
                        help="Cap eval to first N sequences per client")
    parser.add_argument("--max-seqs", type=int, default=None,
                        help="Cap training sequences per client")
    parser.add_argument("--clients", nargs="*", default=None,
                        help="Client ids to include (default: all)")
    args = parser.parse_args()

    main(args)
