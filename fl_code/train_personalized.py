"""Phase 3: Personalised FL — Global TCN + per-client Residual Corrector.

Loads a frozen Global TCN (Phase 2), then trains a local Residual Corrector
for each client with pinball loss.  The Corrector predicts 3-quantile residual
corrections ``E_corr``, producing ``Y_final = Y_pre + E_corr``.

The Corrector input is ``[Y_pre, window context, prev residual]`` per step,
where the window context is a conv-encoded summary of the full input window
(public + load + local channels, shape ``(11 + D_local, input_steps)``).
The whole run output — config.json, personalized_results.json and
``corrector_{cid}.pt`` — is saved per rc type under ``<output-dir>/<rc_type>/``.

Usage::

    python -m fl_code.train_personalized
    python -m fl_code.train_personalized --global-model fl_code/baseline_outputs/nodp/checkpoints/round_020.pt
    python -m fl_code.train_personalized --clients steel_ind_0 tetouan_city_0
    python -m fl_code.train_personalized --max-seqs 5 --epochs 20
    python -m fl_code.train_personalized --output-dir my_run      # custom output root
    python -m fl_code.train_personalized --global-model fl_code/baseline_outputs/dp/checkpoints/round_030.pt 
    --rc-type mlp 
    --output-dir /root/Personalized-Federated-Learning-Power-Forecasting-System/fl_code/personalized_outputs/epsilon-3.5
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
    INPUT_STEPS, PRED_LEN, STRIDE, TRAIN_RATIO,
    CORRECTOR_EPOCHS, CORRECTOR_LR, CORRECTOR_BATCH_SIZE,
)

ROOT = Path(__file__).resolve().parents[1]
CLIENT_CONFIG_PATH = ROOT / "fl_code" / "models" / "client_config.yaml"

# ---------------------------------------------------------------------------
# Lazy dataset variant with local features
# ---------------------------------------------------------------------------


class _LazyWindowsWithLocal(LazySlidingWindowDataset):
    """LazySlidingWindowDataset that appends local features to the input window.

    Returns ``X_full (11 + D_local, input_steps)`` — public channels + load
    from the base class, plus local features at the window positions — so the
    Residual Corrector's window encoder sees the full historical window.
    """

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

        if self.loc_arr is not None:
            # 本地特征可能有 NaN（z-score 后 0 = 训练段均值，客户端内填充）
            loc_win = np.nan_to_num(
                self.loc_arr[start:start + self.input_steps], nan=0.0).T  # (D, T_in)
            X = torch.cat([X, torch.from_numpy(loc_win)], dim=0)          # (11+D, T_in)

        return X, y


# ---------------------------------------------------------------------------
# Pre-compute Y_pre from frozen Global TCN
# ---------------------------------------------------------------------------

@torch.no_grad()
def _precompute_ypre(model, df_norm, seqs, public_cols, local_cols,
                     stride, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream through all training windows, collect (Y_pre, y_true, X_window).

    ``x_window`` keeps the full ``(11 + D_local, input_steps)`` input window
    (public + load + local channels) for the Corrector; the Global TCN only
    consumes the first ``TCNConfig().in_channels`` channels.
    """
    ds = _LazyWindowsWithLocal(
        df_norm, seqs, public_cols, local_cols,
        stride=stride, train=True,
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False, drop_last=False)

    n_global = TCNConfig().in_channels
    ypre_list, y_list, win_list = [], [], []
    for batch in loader:
        X_full = batch[0].to(device)
        ypre_list.append(model(X_full[:, :n_global]).cpu().numpy())
        y_list.append(batch[1].cpu().numpy())
        win_list.append(X_full.cpu().numpy())

    y_pre = np.concatenate(ypre_list, axis=0)
    y_true = np.concatenate(y_list, axis=0)
    x_window = np.concatenate(win_list, axis=0)
    return y_pre, y_true, x_window


# ---------------------------------------------------------------------------
# Corrector training dataset
# ---------------------------------------------------------------------------

class CorrectorDataset(Dataset):
    """Produces (y_pre, residual_history, x_window, target) tuples.

    residual_history[i] = y_true[i-1] - y_pre[i-1]  (previous-window error)
    target[i]           = y_true[i]   - y_pre[i]    (current-window error)

    Index 0 is skipped (no previous residual), so len = N - 1.
    """

    def __init__(self, y_pre: np.ndarray, y_true: np.ndarray,
                 x_window: np.ndarray):
        self.y_pre = torch.from_numpy(y_pre)
        self.y_true = torch.from_numpy(y_true)
        self.x_window = torch.from_numpy(x_window)

    def __len__(self):
        return len(self.y_pre) - 1

    def __getitem__(self, idx):
        i = idx + 1  # offset: need previous window for residual_history
        residual = self.y_true[i - 1] - self.y_pre[i - 1]
        target = self.y_true[i] - self.y_pre[i]

        return self.y_pre[i], residual, self.x_window[i], target


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_corrector_epoch(corrector, loader, optimizer, device):
    """One epoch of Corrector training with pinball loss."""
    corrector.train()
    total_loss = 0.0
    n = 0

    for y_pre, residual, x_window, target in loader:
        y_pre = y_pre.to(device)
        residual = residual.to(device)
        x_window = x_window.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        out = corrector(y_pre, residual, x_window)
        loss = quantile_loss(target, out, corrector.quantiles)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Rolling-forecast evaluation with Corrector
# ---------------------------------------------------------------------------

def _metrics(preds: np.ndarray, actuals: np.ndarray) -> dict:
    """MAE / RMSE / R² / WAPE in normalised space (same as train_eval_utils.evaluate)."""
    mae = float(np.mean(np.abs(preds - actuals)))
    mse = float(np.mean((preds - actuals) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((actuals - preds) ** 2))
    ss_tot = float(np.sum((actuals - actuals.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    denom = float(np.sum(np.abs(actuals)))
    wape = float(np.sum(np.abs(preds - actuals)) / denom) if denom > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "wape": wape}


@torch.no_grad()
def _evaluate_personalized(global_model, corrector, df_norm, seqs,
                           public_cols, local_cols, stride, device):
    """Rolling-forecast eval: Global TCN + Corrector → Y_final.

    Returns two metric dicts (mae/rmse/r2/wape) for the Global TCN P50 vs
    Corrector P50, computed in normalised space (before de-normalisation),
    plus WAPE numerator/denominator components for cross-client aggregation.
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

            # Corrector prediction — window includes local features
            X_rc = X
            if loc_arr is not None:
                # NaN → 0（z-score 后 0 = 训练段均值，客户端内填充）
                loc_win = np.nan_to_num(
                    loc_arr[pos:pos + input_steps], nan=0.0).T      # (D, T_in)
                X_rc = np.concatenate([X_rc, loc_win], axis=0)      # (11+D, T_in)
            X_rc_t = torch.from_numpy(X_rc).unsqueeze(0).to(device)

            # Corrector prediction
            y_pre_np = y_pre.cpu().numpy()
            residual_t = torch.from_numpy(prev_residual).unsqueeze(0).to(device)
            e_corr = corrector(y_pre.unsqueeze(0), residual_t, X_rc_t).squeeze(0)
            y_final = y_pre_np[:, np.newaxis] + e_corr.cpu().numpy()  # (pred_len, 3)

            actual = load[pos + input_steps:pos + input_steps + pred_len]

            all_preds_base.append(y_pre_np)
            all_preds_corr.append(y_final)
            all_actuals.append(actual)

            # Update residual for next step
            prev_residual = actual - y_pre_np
            pos += stride

    if not all_actuals:
        nan_m = {"mae": float("nan"), "rmse": float("nan"),
                 "r2": float("nan"), "wape": float("nan")}
        return nan_m, nan_m, (0.0, 0.0, 0.0)

    actuals = np.concatenate(all_actuals)
    valid = ~np.isnan(actuals)

    base_preds = np.concatenate(all_preds_base)[valid]
    corr_preds = np.concatenate(all_preds_corr)[valid]  # (T, 3)
    actuals = actuals[valid]

    # WAPE 分子/分母分量，供跨客户端全局聚合（Σ|p-a| / Σ|a|）
    sums = (float(np.sum(np.abs(base_preds - actuals))),
            float(np.sum(np.abs(corr_preds[:, 1] - actuals))),
            float(np.sum(np.abs(actuals))))
    return (_metrics(base_preds, actuals),
            _metrics(corr_preds[:, 1], actuals), sums)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_client_data_cached(client_id: str, stride: int,
                             max_seqs: int | None = None,
                             data_dir: str | None = None) -> dict:
    """Load + preprocess, return normalised df and metadata.

    data_dir 非空时走 App 模式：数据源为本地采集目录（app/data），
    列配置仍读 client_config.yaml。
    """
    if data_dir:
        import yaml
        from pathlib import Path as _P
        with open(_P(__file__).resolve().parent
                  / "models" / "client_config.yaml") as f:
            config = yaml.safe_load(f)
        from fl_code.data_utils import _find_client
        dataset_id, client_cfg = _find_client(config, client_id)
        ddir = _P(data_dir)
        csvs = sorted(ddir.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"data-dir 下没有 csv: {ddir}")
        df = pd.read_csv(csvs[0], parse_dates=["datetime"])
        seqs = client_cfg["sequences"]
        public_cols = list(config[dataset_id]["public_features"])
        local_cols = list(config[dataset_id].get("local_features", []))
        # 复刻 load_client_data 的列选择 + one-hot 展开逻辑
        keep = ["datetime"] + seqs + \
            [c for c in public_cols + local_cols if c in df.columns]
        df = df[keep]
        if "category_id" in df.columns:
            cat = df["category_id"].astype(int)
            df["cat_residential"] = (cat == 0).astype(float)
            df["cat_transformer"] = (cat == 1).astype(float)
            df["cat_industrial"] = (cat == 2).astype(float)
            df = df.drop(columns=["category_id"])
        df_norm, _ = preprocess(df, seqs, local_cols)
        return {
            "df_norm": df_norm,
            "seqs": seqs,
            "public_cols": public_cols,
            "local_cols": local_cols,
        }
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

def _latest_global_checkpoint() -> Path | None:
    """Newest non-DP Phase 2 round checkpoint (baseline_outputs/nodp/checkpoints).

    DP global models are never auto-loaded — pass them explicitly via
    --global-model (e.g. fl_code/baseline_outputs/dp/checkpoints/round_020.pt).
    """
    files = sorted((ROOT / "fl_code" / "baseline_outputs" / "nodp" / "checkpoints").glob("round_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    return files[-1] if files else None


def main(args: argparse.Namespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load frozen Global TCN ---
    global_path = (Path(args.global_model) if args.global_model
                   else _latest_global_checkpoint())
    if global_path is None or not global_path.exists():
        raise FileNotFoundError(
            f"Global model not found: {global_path}.  Run Phase 2 first "
            f"(train_baseline.py) — it saves per-round checkpoints to "
            f"fl_code/baseline_outputs/nodp/checkpoints/ (DP models: "
            f"pass --global-model).")
    print(f"Loading Global TCN: {global_path}")
    global_tcn = build_tcn(TCNConfig()).to(device)
    global_tcn.load_state_dict(torch.load(global_path, map_location=device,
                                          weights_only=True))
    global_tcn.eval()
    for p in global_tcn.parameters():
        p.requires_grad = False
    n_global_params = sum(p.numel() for p in global_tcn.parameters())

    # --- Clients ---
    client_ids = _list_clients(args.clients)
    print(f"Clients ({len(client_ids)}): {', '.join(client_ids)}")

    # --- Save run config (model architecture + hyperparameters) ---
    # 每次 rc 运行完全隔离：config / results / 模型全部在 <output-dir>/<rc_type>/
    run_dir = Path(args.output_dir) / args.rc_type
    run_dir.mkdir(parents=True, exist_ok=True)
    corr_cfg = CorrectorConfig(rc_type=args.rc_type)
    corr_dict = {
        "rc_type": args.rc_type,
        "pred_len": corr_cfg.pred_len,
        "quantiles": list(corr_cfg.quantiles),
        "dropout": corr_cfg.dropout,
    }
    if args.rc_type == "mlp":
        corr_dict["hidden_dims"] = list(corr_cfg.hidden_dims)
    elif args.rc_type == "lstm":
        corr_dict["hidden_size"] = corr_cfg.lstm_hidden_size
        corr_dict["num_layers"] = corr_cfg.lstm_num_layers
    else:
        corr_dict["num_channels"] = list(corr_cfg.num_channels)
        corr_dict["kernel_size"] = corr_cfg.kernel_size

    config_json = {
        "script": "fl_code.train_personalized",
        "phase": "Phase 3 — Global TCN + per-client Residual Corrector",
        "global_model": {
            "path": str(global_path),
            "name": "GlobalTCN (frozen)",
            **TCNConfig().to_dict(),
        },
        "corrector": {
            **corr_dict,
            "input": "Y_pre + window_context (x_window 11+D_local → conv-encoded) + residual_history",
            "loss": "pinball (quantile loss)",
            "monotone_quantiles": "softplus increments → e10<=e50<=e90 guaranteed",
            "model_selection": "best epoch by lowest training pinball loss",
        },
        "window_geometry": {
            "input_steps": INPUT_STEPS,
            "pred_len": PRED_LEN,
            "stride": args.stride,
            "train_ratio": TRAIN_RATIO,
        },
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "num_clients": len(client_ids),
            "clients": client_ids,
            "max_seqs": args.max_seqs,
            "eval_seqs": args.eval_seqs,
        },
        "dp": None,   # Phase 3 is pure local training — no DP needed
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2, default=str)
    print(f"Saved run config to {run_dir / 'config.json'}")

    # --- Per-client training ---
    all_results: dict[str, dict] = {}
    t_total0 = time.perf_counter()
    wape_base_num = wape_pers_num = wape_denom = 0.0

    for cid in client_ids:
        print(f"\n{'='*60}")
        print(f"Client: {cid}")
        print(f"{'='*60}")

        data = _load_client_data_cached(cid, args.stride, args.max_seqs,
                                        data_dir=args.data_dir)
        n_seqs = len(data["seqs"])
        local_dim = len(data["local_cols"])

        # ---- Pre-compute Y_pre ----
        t0 = time.perf_counter()
        print(f"  Pre-computing Y_pre ({n_seqs} sequences, local_dim={local_dim})...")
        y_pre, y_true, x_window = _precompute_ypre(
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
        corr_cfg = CorrectorConfig(rc_type=args.rc_type,
                                   local_feat_dim=local_dim)
        corrector = build_corrector(corr_cfg).to(device)
        n_corr_params = sum(p.numel() for p in corrector.parameters())
        print(f"  Corrector: {corr_cfg.rc_type}, {corr_cfg.local_feat_dim} local dims, "
              f"{n_corr_params:,} params")

        # ---- Train Corrector ----
        train_ds = CorrectorDataset(y_pre, y_true, x_window)
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            drop_last=False)
        optimizer = torch.optim.Adam(corrector.parameters(), lr=args.lr)

        t0 = time.perf_counter()
        epoch_losses: list[float] = []
        best_loss = float("inf")
        best_epoch = 0
        best_state = None
        for epoch in range(args.epochs):
            loss = _train_corrector_epoch(corrector, loader, optimizer, device)
            epoch_losses.append(round(loss, 6))
            # keep only the best model (lowest pinball loss)
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone()
                              for k, v in corrector.state_dict().items()}
            if (epoch + 1) % max(1, args.epochs // 5) == 0:
                print(f"  Epoch {epoch + 1:3d}/{args.epochs}  loss={loss:.6f}")
        train_time = time.perf_counter() - t0
        print(f"  Training: {train_time:.1f}s  "
              f"(best epoch {best_epoch}, loss={best_loss:.6f})")

        # ---- Evaluate (on the best-epoch model) ----
        corrector.load_state_dict(best_state)
        print(f"  Evaluating rolling forecast (best epoch {best_epoch}) ...")
        t0 = time.perf_counter()
        eval_seqs = data["seqs"]
        if args.eval_seqs and len(eval_seqs) > args.eval_seqs:
            eval_seqs = eval_seqs[:args.eval_seqs]

        m_base, m_pers, wape_sums = _evaluate_personalized(
            global_tcn, corrector, data["df_norm"],
            eval_seqs, data["public_cols"], data["local_cols"],
            args.stride, device,
        )
        wape_base_num += wape_sums[0]
        wape_pers_num += wape_sums[1]
        wape_denom += wape_sums[2]
        eval_time = time.perf_counter() - t0
        mae_base, mae_corr = m_base["mae"], m_pers["mae"]
        print(f"  Eval: {eval_time:.1f}s")
        print(f"  Baseline MAE (Y_pre):   {mae_base:.4f}  "
              f"RMSE={m_base['rmse']:.4f}  R^2={m_base['r2']:.4f}")
        print(f"  Personal MAE (Y_final): {mae_corr:.4f}  "
              f"RMSE={m_pers['rmse']:.4f}  R^2={m_pers['r2']:.4f}")
        gain = None
        if not np.isnan(mae_base) and not np.isnan(mae_corr):
            gain = (mae_base - mae_corr) / mae_base * 100
            print(f"  Improvement: {gain:+.1f}%")

        # ---- Save Corrector (best epoch only) — per-rc_type subfolder ----
        torch.save(best_state, run_dir / f"corrector_{cid}.pt")

        all_results[cid] = {
            "mae_baseline": m_base["mae"],
            "rmse_baseline": m_base["rmse"],
            "r2_baseline": m_base["r2"],
            "mae_personalized": m_pers["mae"],
            "rmse_personalized": m_pers["rmse"],
            "r2_personalized": m_pers["r2"],
            "wape_baseline": m_base["wape"],
            "wape_personalized": m_pers["wape"],
            "improvement_mae_pct": round(gain, 2) if gain is not None else None,
            "n_windows": n_windows,
            "n_seqs": n_seqs,
            "local_dim": local_dim,
            "corrector_type": corr_cfg.rc_type,
            "corrector_params": n_corr_params,
            "train_time_s": round(train_time, 1),
            "eval_time_s": round(eval_time, 1),
            "epoch_losses": epoch_losses,
            "best_epoch": best_epoch,
            "best_loss": round(best_loss, 6),
        }

    # --- Aggregate metrics (aligned with baseline_history.json) ---
    total_train_time = time.perf_counter() - t_total0
    client_metrics = {}
    valid_base = [r["mae_baseline"] for r in all_results.values()
                  if not np.isnan(r["mae_baseline"])]
    valid_pers = [r["mae_personalized"] for r in all_results.values()
                  if not np.isnan(r["mae_personalized"])]
    gains = [r["improvement_mae_pct"] for r in all_results.values()
             if r["improvement_mae_pct"] is not None]
    avg_base = float(np.mean(valid_base)) if valid_base else float("nan")
    avg_pers = float(np.mean(valid_pers)) if valid_pers else float("nan")
    avg_gain = float(np.mean(gains)) if gains else float("nan")
    for cid, r in all_results.items():
        client_metrics[cid] = {
            "mae": r["mae_personalized"],
            "rmse": r["rmse_personalized"],
            "r2": r["r2_personalized"],
            "n_train": r["n_windows"],
        }
    final_metrics = {
        "avg_mae_baseline": avg_base,
        "avg_mae_personalized": avg_pers,
        "avg_improvement_mae_pct": (round(avg_gain, 2)
                                    if not np.isnan(avg_gain) else None),
        "wape_baseline": (float(wape_base_num / wape_denom)
                          if wape_denom > 0 else float("nan")),
        "wape_personalized": (float(wape_pers_num / wape_denom)
                              if wape_denom > 0 else float("nan")),
        "client_metrics": client_metrics,
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
    print("-" * 52)
    print(f"  {'Average':20s}  {avg_base:10.4f}  {avg_pers:10.4f}  "
          f"{avg_gain:+8.2f}%")

    # --- Save results ---
    with open(run_dir / "personalized_results.json", "w") as f:
        json.dump({
            "args": {k: str(v) for k, v in vars(args).items()},
            "global_model": str(global_path),
            "num_clients": len(client_ids),
            "global_model_params": n_global_params,
            "training_time_s": round(total_train_time, 1),
            "final_metrics": final_metrics,
            "dp": None,   # Phase 3 is pure local training — no DP needed
            "results": all_results,
        }, f, indent=2, default=str)

    # --- Update config with per-client local feature dims ---
    config_json["corrector"]["local_feat_dim_per_client"] = {
        cid: r["local_dim"] for cid, r in all_results.items()
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2, default=str)

    print(f"\nOutputs saved to {run_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3: Personalised FL — Global TCN + Residual Corrector",
    )
    parser.add_argument("--global-model", type=str, default=None,
                        help="Path to frozen Global TCN checkpoint (default: newest "
                             "non-DP fl_code/baseline_outputs/nodp/checkpoints/"
                             "round_*.pt; DP models must be passed explicitly)")
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
    parser.add_argument("--output-dir", type=str,
                        default=str(ROOT / "fl_code" / "personalized_outputs"),
                        help="Output root directory (default: fl_code/personalized_outputs)")
    parser.add_argument("--rc-type", type=str, default="mlp",
                        choices=["mlp", "lstm", "tcn"],
                        help="Residual Corrector architecture (default: mlp — "
                             "simplest, fastest; switch to tcn/lstm if needed)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="App 模式：本地采集数据目录（默认用 data/processed）")
    args = parser.parse_args()

    main(args)
