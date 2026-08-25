"""Standalone evaluation of trained Global TCN / Corrector models in
de-normalised (raw physical) units.

Phase 2/3 training scripts compute validation metrics in normalised space.
This script re-runs the same rolling-forecast protocol (test portion, stride
= pred_len) but de-normalises **per sequence** — each sequence has its own
log1p + z-score parameters from ``fl_code.data_utils.preprocess`` — before
computing MAE / RMSE / R² / WAPE in original kWh units.

Three variants are evaluated on all clients:

  - ``nodp``   — no-DP Global TCN point forecast (Y_pre)
  - ``dp``     — DP Global TCN point forecast (Y_pre)
  - ``dp+rc``  — DP Global TCN + per-client Residual Corrector (Y_final P50)

Variants whose models are unavailable are skipped (with a warning).  The
``dp+rc`` pairing assumes the Corrector was trained against the DP Global
TCN, as recorded in ``personalized_outputs/<rc_type>/config.json``.

Outputs:
  - JSON metrics (default ``fl_code/denorm_eval/denorm_metrics.json``)
  - ``fig_denorm_per_client.png`` — per-client WAPE (%), 3 variants
    (the aggregate comparison lives in make_figures' standard figure set)

Usage::

    python -m fl_code.validate_denorm
    python -m fl_code.validate_denorm --nodp-global <path> --dp-global <path>
    python -m fl_code.validate_denorm --rc-type tcn --rc-dir <dir>
    python -m fl_code.validate_denorm --clients steel_ind_0 \\
        --output-json /tmp/denorm.json --fig-dir /tmp/figs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from fl_code.config import INPUT_STEPS, PRED_LEN, STRIDE, TRAIN_RATIO
from fl_code.data_utils import load_client_data, preprocess
from fl_code.models import (
    TCNConfig, CorrectorConfig,
    build_tcn, build_corrector,
)

ROOT = Path(__file__).resolve().parents[1]
CLIENT_CONFIG_PATH = ROOT / "fl_code" / "models" / "client_config.yaml"
BASELINE_DIR = ROOT / "fl_code" / "baseline_outputs"
PERSONALIZED_DIR = ROOT / "fl_code" / "personalized_outputs"
DEFAULT_JSON = ROOT / "fl_code" / "denorm_eval" / "denorm_metrics.json"
DEFAULT_FIG_DIR = ROOT / "fl_code" / "figures"

VARIANTS = ("nodp", "dp", "dp+rc")
STAGE_COLORS = {"nodp": "#d62728", "dp": "#1f77b4", "dp+rc": "#2ca02c"}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _latest_checkpoint(variant: str) -> Path | None:
    files = sorted((BASELINE_DIR / variant / "checkpoints").glob("round_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    return files[-1] if files else None


def _resolve_global(variant: str, explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"{variant} global model not found: {p}")
        return p
    p = _latest_checkpoint(variant)
    if p is None:
        print(f"WARNING: no {variant} checkpoints in "
              f"{BASELINE_DIR / variant / 'checkpoints'} — variant skipped")
    return p


def _load_tcn(path: Path, device: str) -> torch.nn.Module:
    model = build_tcn(TCNConfig()).to(device)
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def _load_corrector(path: Path, rc_type: str, local_dim: int,
                    device: str) -> torch.nn.Module:
    corrector = build_corrector(CorrectorConfig(
        rc_type=rc_type, local_feat_dim=local_dim)).to(device)
    corrector.load_state_dict(
        torch.load(path, map_location=device, weights_only=True))
    corrector.eval()
    return corrector


# ---------------------------------------------------------------------------
# Rolling-forecast evaluation (de-normalised)
# ---------------------------------------------------------------------------

@torch.no_grad()
def rolling_eval_denorm(model: torch.nn.Module,
                        corrector: torch.nn.Module | None,
                        df_norm, seqs: list[str], public_cols: list[str],
                        local_cols: list[str], params: dict,
                        stride: int = STRIDE, device: str = "cpu"):
    """Rolling-forecast eval over the test portion of every sequence.

    Identical protocol to ``train_personalized._evaluate_personalized``
    (input window ``INPUT_STEPS`` / horizon ``PRED_LEN`` / prev-residual
    chain for the Corrector), but predictions and actuals are
    de-normalised **per sequence** with ``params[seq]`` (log1p + z-score)
    and concatenated in raw kWh units.

    Parameters
    ----------
    model : nn.Module
        Global TCN point-forecast model.
    corrector : nn.Module or None
        Residual Corrector (P50 output) or None for the plain baseline.
    df_norm : DataFrame
        Normalised client data (from :func:`~fl_code.data_utils.preprocess`).
    seqs, public_cols, local_cols : list[str]
        Column names; ``params[seq]`` holds each sequence's de-normalisation
        parameters.
    stride : int
        Rolling step (default ``STRIDE`` = pred_len → gap-free coverage).

    Returns
    -------
    preds : np.ndarray
        1-D array of raw-unit predictions (Corrector: P50).
    actuals : np.ndarray
        1-D array of raw-unit actuals, aligned with *preds*.
    """
    input_steps, pred_len = INPUT_STEPS, PRED_LEN
    pub_arr = df_norm[public_cols].values.astype(np.float32)
    loc_arr = (df_norm[local_cols].values.astype(np.float32)
               if local_cols else None)

    all_preds, all_actuals = [], []

    for s in seqs:
        load = df_norm[s].values.astype(np.float32)
        f = df_norm[s].first_valid_index()
        l = df_norm[s].last_valid_index()
        if f is None or l is None:
            continue

        p = params[s]
        valid_len = l - f + 1
        split = f + int(valid_len * TRAIN_RATIO)

        prev_residual = np.zeros(pred_len, dtype=np.float32)
        pos = split
        preds_norm, actuals_norm = [], []

        while pos + input_steps + pred_len <= l + 1:
            X_pub = pub_arr[pos:pos + input_steps].T
            X_load = load[pos:pos + input_steps][np.newaxis, :]
            X = np.concatenate([X_pub, X_load], axis=0)
            X_t = torch.from_numpy(X).unsqueeze(0).to(device)
            y_pre = model(X_t).squeeze(0).cpu().numpy()       # (pred_len,)

            if corrector is not None:
                X_rc = X
                if loc_arr is not None:
                    # NaN → 0（z-score 后 0 = 训练段均值，客户端内填充）
                    loc_win = np.nan_to_num(
                        loc_arr[pos:pos + input_steps], nan=0.0).T  # (D, T_in)
                    X_rc = np.concatenate([X_rc, loc_win], axis=0)
                e = corrector(
                    torch.from_numpy(y_pre).unsqueeze(0).to(device),
                    torch.from_numpy(prev_residual).unsqueeze(0).to(device),
                    torch.from_numpy(X_rc).unsqueeze(0).to(device),
                ).squeeze(0).cpu().numpy()                     # (pred_len, 3)
                y_pred = (y_pre[:, np.newaxis] + e)[:, 1]      # P50
            else:
                y_pred = y_pre

            actual = load[pos + input_steps:pos + input_steps + pred_len]

            preds_norm.append(y_pred)
            actuals_norm.append(actual)

            prev_residual = actual - y_pre
            pos += stride

        if not preds_norm:
            continue

        preds_s = np.concatenate(preds_norm)
        actuals_s = np.concatenate(actuals_norm)
        # De-normalise (load sequences always use log1p + z-score)
        all_preds.append(np.expm1(preds_s * p["std"] + p["mean"]))
        all_actuals.append(np.expm1(actuals_s * p["std"] + p["mean"]))

    if not all_preds:
        return np.array([]), np.array([])
    return np.concatenate(all_preds), np.concatenate(all_actuals)


# ---------------------------------------------------------------------------
# Metrics (raw units)
# ---------------------------------------------------------------------------

def _metrics(actuals: np.ndarray, preds: np.ndarray) -> dict:
    """MAE / RMSE / R² / WAPE on valid (non-NaN) pairs, raw kWh units."""
    valid = ~np.isnan(actuals) & ~np.isnan(preds)
    a, p = actuals[valid], preds[valid]
    if len(a) == 0:
        return {"mae": float("nan"), "rmse": float("nan"),
                "r2": float("nan"), "wape": float("nan"), "n": 0}
    mae = float(np.mean(np.abs(p - a)))
    rmse = float(np.sqrt(np.mean((p - a) ** 2)))
    ss_res = float(np.sum((a - p) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    denom = float(np.sum(np.abs(a)))
    wape = float(np.sum(np.abs(p - a)) / denom) if denom > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "wape": wape, "n": int(len(a))}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_client(client_id: str, max_seqs: int | None) -> dict:
    """Load + preprocess a client; keep the per-sequence de-norm params."""
    df, info = load_client_data(client_id)
    feat_names = set(info["public_features"] + info["local_features"])
    seqs = [c for c in df.columns if c not in feat_names and c != "datetime"]
    if max_seqs and len(seqs) > max_seqs:
        seqs = seqs[:max_seqs]

    df_norm, params = preprocess(df, seqs, info["local_features"])
    return {
        "df_norm": df_norm,
        "seqs": seqs,
        "params": params,
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
# Visualization
# ---------------------------------------------------------------------------

def _setup_cn_font() -> bool:
    """Try to register a CJK font; return True when Chinese can be rendered."""
    import matplotlib
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
                 "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei", "PingFang SC"):
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    print("WARNING: no CJK font found — falling back to English labels")
    return False


def _labels(cn: bool) -> dict[str, str]:
    if cn:
        return {
            "note": "注：指标为反归一化后（原始单位 kWh）",
            "v_nodp": "nodp (Y_pre)",
            "v_dp": "dp (Y_pre)",
            "v_dp+rc": "dp+rc (Y_final P50)",
            "metric_wape": "WAPE (%)",
            "fig_per_client": "反归一化后每客户端 WAPE 对比（原始单位 kWh）",
            "clients": "客户端",
        }
    return {
        "note": "Note: metrics computed after de-normalisation (raw kWh units)",
        "v_nodp": "nodp (Y_pre)",
        "v_dp": "dp (Y_pre)",
        "v_dp+rc": "dp+rc (Y_final P50)",
        "metric_wape": "WAPE (%)",
        "fig_per_client": "Per-client de-normalised WAPE (raw kWh)",
        "clients": "Client",
    }


def _finish(fig, L: dict) -> None:
    fig.text(0.99, 0.01, L["note"], ha="right", va="bottom",
             fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def _fig_per_client(per_client: dict, L: dict, plt) -> None:
    """Single panel: per-client WAPE (%), 3 variants per client."""
    cids = list(per_client)
    fig, ax = plt.subplots(figsize=(11.5, 3.8))
    x = np.arange(len(cids))
    w = 0.30                     # bars nearly touching (thin white seams)

    maxv = 0.0
    for i, v in enumerate(VARIANTS):
        vals = []
        for cid in cids:
            m = per_client[cid].get(v)
            if m is None or m.get("wape") is None:
                vals.append(np.nan)
            else:
                maxv = max(maxv, m["wape"] * 100)
                vals.append(m["wape"] * 100)
                ax.text(x[cids.index(cid)] + (i - 1) * w,
                        m["wape"] * 100 + 0.5, f"{m['wape']*100:.1f}",
                        ha="center", va="bottom", fontsize=5.5,
                        style="italic", color="#333333")
        ax.bar(x + (i - 1) * w, vals, w, label=L["v_" + v],
               color=STAGE_COLORS[v], edgecolor="white", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cids, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(L["metric_wape"])
    ax.legend(loc="upper center", ncol=3, fontsize=8)
    ax.grid(axis="y", zorder=2.5, color="#b0b0b0", lw=0.6, alpha=0.9)
    ax.set_ylim(0, maxv * 1.15)

    fig.suptitle(L["fig_per_client"], fontsize=12)
    _finish(fig, L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Resolve global models ---
    nodp_meta = None
    if args.nodp_json:
        with open(args.nodp_json) as f:
            nodp_meta = json.load(f)
        print(f"nodp 指标从 {args.nodp_json} 载入（跳过 nodp 评估）")
    nodp_path = (None if nodp_meta else
                 _resolve_global("nodp", args.nodp_global))
    dp_path = _resolve_global("dp", args.dp_global)
    if nodp_path is None and dp_path is None and nodp_meta is None:
        raise SystemExit("No global model available — "
                         "pass --nodp-global / --dp-global / --nodp-json")
    print(f"Global models:  nodp={nodp_path or ('--nodp-json' if nodp_meta else 'SKIPPED')}")
    print(f"                dp  ={dp_path or 'SKIPPED'}")

    nodp_model = _load_tcn(nodp_path, device) if nodp_path else None
    dp_model = _load_tcn(dp_path, device) if dp_path else None

    rc_dir = (Path(args.rc_dir) if args.rc_dir
              else PERSONALIZED_DIR / args.rc_type)
    print(f"Corrector dir:  {rc_dir} (type={args.rc_type})")

    # --- Clients ---
    client_ids = _list_clients(args.clients)
    print(f"Clients ({len(client_ids)}): {', '.join(client_ids)}")

    per_client: dict[str, dict] = {}
    agg_sums: dict[str, list[float]] = {v: [0.0, 0.0] for v in VARIANTS}
    agg_avgs: dict[str, dict] = {v: {"mae": [], "rmse": [], "r2": []}
                                 for v in VARIANTS}

    for cid in client_ids:
        data = _load_client(cid, args.max_seqs)
        seqs = data["seqs"]
        if args.eval_seqs and len(seqs) > args.eval_seqs:
            seqs = seqs[:args.eval_seqs]
        print(f"\nClient {cid}: {len(seqs)} sequences, "
              f"{len(data['local_cols'])} local features")

        corrector = None
        if dp_model is not None:
            ckpt = rc_dir / f"corrector_{cid}.pt"
            if ckpt.exists():
                corrector = _load_corrector(
                    ckpt, args.rc_type, len(data["local_cols"]), device)
            else:
                print(f"  WARNING: {ckpt} not found — dp+rc skipped for {cid}")

        client_res: dict = {}
        for variant in VARIANTS:
            if variant == "nodp" and nodp_meta is not None:
                # 复用预计算 nodp 指标：同一模型、同一协议，结果逐字节一致
                client_res[variant] = (
                    nodp_meta["per_client"].get(cid) or {}).get("nodp")
                continue
            if variant == "nodp":
                model = nodp_model
            elif variant == "dp":
                model = dp_model
            else:
                model = dp_model if corrector is not None else None
            if model is None:
                client_res[variant] = None
                continue

            preds, actuals = rolling_eval_denorm(
                model, corrector if variant == "dp+rc" else None,
                data["df_norm"], seqs, data["public_cols"],
                data["local_cols"], data["params"], args.stride, device)
            met = _metrics(actuals, preds)
            client_res[variant] = met
            print(f"  [{variant:5s}] n={met['n']:6d}  "
                  f"MAE={met['mae']:.3f}  RMSE={met['rmse']:.3f}  "
                  f"R2={met['r2']:.3f}  WAPE={met['wape'] * 100:.2f}%")

            valid = ~np.isnan(actuals) & ~np.isnan(preds)
            if valid.any():
                a, p = actuals[valid], preds[valid]
                agg_sums[variant][0] += float(np.sum(np.abs(p - a)))
                agg_sums[variant][1] += float(np.sum(np.abs(a)))
                agg_avgs[variant]["mae"].append(met["mae"])
                agg_avgs[variant]["rmse"].append(met["rmse"])
                if not np.isnan(met["r2"]):
                    agg_avgs[variant]["r2"].append(met["r2"])

        per_client[cid] = client_res

    # --- Aggregate (same convention as train scripts) ---
    aggregate: dict[str, dict] = {}
    for v in VARIANTS:
        s_num, s_den = agg_sums[v]
        mae = agg_avgs[v]["mae"]
        aggregate[v] = {
            "wape": float(s_num / s_den) if s_den > 0 else float("nan"),
            "avg_mae": float(np.mean(mae)) if mae else float("nan"),
            "avg_rmse": (float(np.mean(agg_avgs[v]["rmse"]))
                         if agg_avgs[v]["rmse"] else float("nan")),
            "avg_r2": (float(np.mean(agg_avgs[v]["r2"]))
                       if agg_avgs[v]["r2"] else float("nan")),
            "num_clients": len(mae),
        }
    if nodp_meta is not None:
        aggregate["nodp"] = nodp_meta["aggregate"]["nodp"]

    # --- Summary table ---
    print(f"\n{'=' * 78}")
    print("Summary - de-normalised metrics (raw kWh units)")
    print(f"{'=' * 78}")
    for v in VARIANTS:
        cells = []
        for cid in client_ids:
            m = per_client[cid].get(v)
            cells.append(f"{cid}={m['wape'] * 100:.1f}%"
                         if m else f"{cid}=n/a")
        print(f"[{v:5s}] " + "  ".join(cells))
    print("\nAggregate:")
    for v in VARIANTS:
        a = aggregate[v]
        print(f"  [{v:5s}] WAPE={a['wape'] * 100:.2f}%  "
              f"avg_MAE={a['avg_mae']:.3f}  avg_RMSE={a['avg_rmse']:.3f}  "
              f"avg_R2={a['avg_r2']:.3f}  (clients={a['num_clients']})")

    # --- Save JSON ---
    out_json = Path(args.output_json) if args.output_json else DEFAULT_JSON
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "script": "fl_code.validate_denorm",
            "args": {k: str(v) for k, v in vars(args).items()},
            "models": {"nodp": (str(nodp_path) if nodp_path else
                                (str(args.nodp_json) if nodp_meta else None)),
                       "dp": str(dp_path) if dp_path else None,
                       "rc_dir": str(rc_dir),
                       "rc_type": args.rc_type},
            "note": "metrics computed after de-normalisation (raw kWh units)",
            "per_client": per_client,
            "aggregate": aggregate,
        }, f, indent=2, default=str)
    print(f"\nSaved JSON: {out_json}")

    # --- Save figures ---
    import matplotlib
    matplotlib.use("TkAgg" if args.show else "Agg")
    import matplotlib.pyplot as plt

    cn = _setup_cn_font()
    L = _labels(cn)
    fig_dir = Path(args.fig_dir) if args.fig_dir else DEFAULT_FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    _fig_per_client(per_client, L, plt)
    p = fig_dir / "fig_denorm_per_client.png"
    plt.gcf().savefig(p, dpi=150)
    plt.close()
    print(f"Saved: {p}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained Global TCN / Corrector models in "
                    "de-normalised (raw kWh) units")
    parser.add_argument("--nodp-global", type=str, default=None,
                        help="No-DP Global TCN checkpoint (default: newest "
                             "fl_code/baseline_outputs/nodp/checkpoints/round_*.pt)")
    parser.add_argument("--nodp-json", type=str, default=None,
                        help="Precomputed metrics JSON (validate_denorm output) "
                             "whose per_client/aggregate nodp values are reused "
                             "verbatim, skipping the nodp rolling evaluation "
                             "(dedup; mutually redundant with --nodp-global)")
    parser.add_argument("--dp-global", type=str, default=None,
                        help="DP Global TCN checkpoint (default: newest "
                             "fl_code/baseline_outputs/dp/checkpoints/round_*.pt)")
    parser.add_argument("--rc-type", type=str, default="mlp",
                        choices=["mlp", "lstm", "tcn"],
                        help="Residual Corrector architecture (default: mlp)")
    parser.add_argument("--rc-dir", type=str, default=None,
                        help="Directory containing corrector_{cid}.pt files "
                             "(default: fl_code/personalized_outputs/<rc-type>)")
    parser.add_argument("--clients", nargs="*", default=None,
                        help="Client ids to include (default: all)")
    parser.add_argument("--stride", type=int, default=STRIDE,
                        help=f"Rolling-forecast stride (default: {STRIDE})")
    parser.add_argument("--eval-seqs", type=int, default=None,
                        help="Cap eval to first N sequences per client")
    parser.add_argument("--max-seqs", type=int, default=None,
                        help="Cap data sequences per client")
    parser.add_argument("--output-json", type=str, default=None,
                        help=f"JSON metrics output path "
                             f"(default: {DEFAULT_JSON})")
    parser.add_argument("--fig-dir", type=str, default=None,
                        help=f"Figure output directory (default: {DEFAULT_FIG_DIR})")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda/cpu (default: auto)")
    parser.add_argument("--show", action="store_true",
                        help="Open figure windows (GUI)")
    args = parser.parse_args()
    main(args)
