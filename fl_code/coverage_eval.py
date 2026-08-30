"""P10-P90 interval coverage evaluation for the DP + best-RC pipeline.

For each privacy budget in the sweep (epsilon in {0.5, 1.5, 3.5, 5.5, 7.5})
this script loads the **final-round DP Global TCN** checkpoint together with
each client's best-performing Residual Corrector (architecture chosen per
client by the lowest ``dp+rc`` WAPE recorded in
``fl_code/analysis/epsilon-<eps>/denorm_metrics_<rc>.json``), runs the same
rolling-forecast protocol as training (input 144 steps / horizon 6, gap-free
stride 6, prev-residual chain for the Corrector), and measures how often the
actual load falls inside the predicted [P10, P90] interval:

    coverage = mean(P10 <= y_true <= P90)      target ~ 80%

All interval bounds and actuals are de-normalised per sequence (log1p +
z-score) before comparison, so numbers are in raw kWh units.

Recording budget: for large clients the test region holds 10^5+ windows; the
prev-residual chain is still evaluated at the full gap-free stride (identical
inputs to training/eval), but interval statistics are **recorded** only every
k-th window so that each client contributes at most ``--max-windows`` recorded
points (k chosen per client from a cheap window count).

Outputs
-------
- ``fl_code/analysis/coverage/coverage_results.json``
- ``fl_code/analysis/figs/fig_coverage_epsilon.png``  (aggregate coverage vs
  epsilon + per-client bars, 80% target line)

Usage::

    python -m fl_code.coverage_eval
    python -m fl_code.coverage_eval --eps 0.5 7.5 --max-windows 2000
    python -m fl_code.coverage_eval --clients steel_ind_0 lcl_res_0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from fl_code.config import INPUT_STEPS, PRED_LEN, STRIDE, TRAIN_RATIO
from fl_code.data_utils import load_client_data, preprocess
from fl_code.models import TCNConfig, CorrectorConfig, build_tcn, build_corrector

ROOT = Path(__file__).resolve().parents[1]
CLIENT_CONFIG_PATH = ROOT / "fl_code" / "models" / "client_config.yaml"
BASELINE_DIR = ROOT / "fl_code" / "baseline_outputs"
ANALYSIS_DIR = ROOT / "fl_code" / "analysis"
PERSONALIZED_DIR = ROOT / "fl_code" / "personalized_outputs"
DEFAULT_JSON = ROOT / "fl_code" / "analysis" / "coverage" / "coverage_results.json"
DEFAULT_FIG_DIR = ROOT / "fl_code" / "analysis" / "figs"

EPS_LIST = ("0.5", "1.5", "3.5", "5.5", "7.5")
RC_TYPES = ("mlp", "lstm", "tcn")
TARGET_COVERAGE = 0.80

# matplotlib palette (matches make_figures convention)
EPS_COLORS = {
    "0.5": "#9ecae1", "1.5": "#6baed6", "3.5": "#4292c6",
    "5.5": "#2171b5", "7.5": "#08519c",
}


# ---------------------------------------------------------------------------
# Checkpoints & best-RC selection
# ---------------------------------------------------------------------------

def resolve_round_ckpt(eps: str | None, round_no: int) -> Path:
    """Checkpoint for a DP epsilon sweep run (``eps=None`` → no-dp run)."""
    run_dir = BASELINE_DIR / ("no-dp" if eps is None else f"dp/epsilon-{eps}")
    exact = run_dir / "checkpoints" / f"round_{round_no:03d}.pt"
    if exact.exists():
        return exact
    files = sorted(run_dir.glob("checkpoints/round_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise FileNotFoundError(f"no checkpoints under {run_dir / 'checkpoints'}")
    return files[-1]


def best_rc_type(eps: str, cid: str) -> tuple[str, float]:
    """RC architecture with the lowest dp+rc WAPE for (eps, client)."""
    best_type, best_wape = "mlp", math.inf
    for rc in RC_TYPES:
        p = ANALYSIS_DIR / f"epsilon-{eps}" / f"denorm_metrics_{rc}.json"
        if not p.exists():
            continue
        with open(p) as f:
            met = json.load(f)
        m = (met.get("per_client", {}).get(cid) or {}).get("dp+rc")
        if m and m.get("wape") is not None and m["wape"] < best_wape:
            best_type, best_wape = rc, float(m["wape"])
    if math.isinf(best_wape):
        print(f"  WARNING: no dp+rc metrics for {cid} @ eps={eps} — default rc=mlp")
        best_wape = float("nan")
    return best_type, best_wape


def load_tcn(path: Path, device: str) -> torch.nn.Module:
    model = build_tcn(TCNConfig()).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def load_corrector(path: Path, rc_type: str, local_dim: int,
                   device: str) -> torch.nn.Module:
    corrector = build_corrector(CorrectorConfig(
        rc_type=rc_type, local_feat_dim=local_dim)).to(device)
    corrector.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    corrector.eval()
    return corrector


def list_clients(whitelist: list[str] | None = None) -> list[str]:
    with open(CLIENT_CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    ids = [cid for ds in config.values() for cid in ds["clients"]]
    return [c for c in ids if not whitelist or c in whitelist]


# ---------------------------------------------------------------------------
# Rolling-forecast interval evaluation
# ---------------------------------------------------------------------------

def _denorm(x: np.ndarray, p: dict) -> np.ndarray:
    return np.expm1(x * p["std"] + p["mean"])


@torch.no_grad()
def rolling_eval_intervals(model: torch.nn.Module, corrector: torch.nn.Module,
                           df_norm, seqs: list[str], public_cols: list[str],
                           local_cols: list[str], params: dict,
                           max_windows: int, device: str,
                           max_chain_windows: int = 20000) -> dict:
    """Gap-free rolling forecast over the test portion; record P10/P50/P90.

    Global-TCN forwards are batched per sequence; the Corrector prev-residual
    chain stays sequential and identical to the training/eval protocol
    (stride = PRED_LEN, residual = actual - y_pre of the previous window,
    held at the last valid value when the actual contains NaN).

    Windows with any NaN in the load segment (input + horizon) or in the
    public-feature input are skipped (same NaN policy as
    ``data_utils.make_sliding_windows``).  At most ``max_windows`` windows
    per client are **recorded**, uniformly spread over the test region.
    """
    input_steps, pred_len, stride = INPUT_STEPS, PRED_LEN, STRIDE
    pub_arr = df_norm[public_cols].values.astype(np.float32)
    loc_arr = (df_norm[local_cols].values.astype(np.float32)
               if local_cols else None)

    # --- pass 1: window positions per sequence (cheap, no model) ---
    # sequences are taken in order until the chain-window budget is exhausted
    # (large clients such as eld_ind_0 hold 4e5+ test windows; the first
    # sequences already give a representative, protocol-faithful sample)
    seq_windows: dict[str, list[int]] = {}
    total = 0
    for s in seqs:
        if total >= max_chain_windows:
            break
        f = df_norm[s].first_valid_index()
        l = df_norm[s].last_valid_index()
        if f is None or l is None:
            continue
        valid_len = l - f + 1
        split = f + int(valid_len * TRAIN_RATIO)
        load = df_norm[s].values.astype(np.float32)
        pos_list = []
        pos = split
        while pos + input_steps + pred_len <= l + 1:
            seg = load[pos:pos + input_steps + pred_len]
            if not np.isnan(seg).any() and not np.isnan(pub_arr[pos:pos + input_steps]).any():
                pos_list.append(pos)
            pos += stride
        if total + len(pos_list) > max_chain_windows and pos_list:
            pos_list = pos_list[:max(1, max_chain_windows - total)]
        if pos_list:
            seq_windows[s] = pos_list
            total += len(pos_list)
    if total == 0:
        return {"n": 0}

    k = max(1, math.ceil(total / max_windows))   # record every k-th window

    # per-sequence raw load range over the test region.  Normalisation MUST
    # happen per sequence: meters inside one client can differ by two orders
    # of magnitude (eld_ind: 67 kWh ~ 47,145 kWh), so a client-wide pooled
    # range would crush the small meters' PINAW to ~0.
    seq_range: dict[str, float] = {}
    for s in seqs:
        f = df_norm[s].first_valid_index()
        l = df_norm[s].last_valid_index()
        if f is None or l is None:
            continue
        split = f + int((l - f + 1) * TRAIN_RATIO)
        seg = df_norm[s].values[split:l + 1].astype(np.float32)
        seg = seg[~np.isnan(seg)]
        if seg.size:
            raw_seg = _denorm(seg, params[s])
            seq_range[s] = float(raw_seg.max() - raw_seg.min())

    lo_all, hi_all, act_all = [], [], []
    rel_w_all = []   # per recorded window: width / own sequence's raw range
    recorded = 0
    g_idx = 0                                     # global window counter

    for s, pos_list in seq_windows.items():
        load = df_norm[s].values.astype(np.float32)
        p = params[s]
        rec_local = set(range((-(g_idx) % k), len(pos_list), k))  # keeps a global cadence
        g_idx += len(pos_list)

        # batched Global TCN forward over all windows of this sequence
        X = np.stack([
            np.concatenate([pub_arr[i:i + input_steps].T,
                            load[i:i + input_steps][np.newaxis, :]], axis=0)
            for i in pos_list])
        y_pre_all = []
        for c0 in range(0, len(X), 4096):
            xb = torch.from_numpy(X[c0:c0 + 4096]).to(device)
            y_pre_all.append(model(xb).cpu().numpy())
        y_pre_all = np.concatenate(y_pre_all)     # (n, pred_len)

        prev_residual = np.zeros(pred_len, dtype=np.float32)
        for j, pos in enumerate(pos_list):
            y_pre = y_pre_all[j]
            X_rc = X[j]
            if loc_arr is not None:
                # window context: local features over the INPUT window
                # (NaN -> 0 = training-portion mean, same as validate_denorm)
                loc_win = np.nan_to_num(
                    loc_arr[pos:pos + input_steps], nan=0.0).T  # (D, T_in)
                X_rc = np.concatenate([X_rc, loc_win], axis=0)
            e = corrector(
                torch.from_numpy(y_pre).unsqueeze(0).to(device),
                torch.from_numpy(prev_residual).unsqueeze(0).to(device),
                torch.from_numpy(X_rc).unsqueeze(0).to(device),
            ).squeeze(0).cpu().numpy()            # (pred_len, 3)
            actual = load[pos + input_steps:pos + input_steps + pred_len]
            if np.isnan(actual).any():
                prev_residual = np.where(np.isnan(actual), prev_residual,
                                         actual - y_pre)
            else:
                prev_residual = actual - y_pre

            if j not in rec_local:
                continue
            a = _denorm(actual, p)
            lo = _denorm(y_pre + e[:, 0], p)
            hi = _denorm(y_pre + e[:, 2], p)
            valid = ~np.isnan(a)
            lo_all.append(lo[valid])
            hi_all.append(hi[valid])
            act_all.append(a[valid])
            rng = seq_range.get(s, 0.0)
            if rng > 0:
                rel_w_all.append((hi[valid] - lo[valid]) / rng)
            recorded += 1

    lo_all = np.concatenate(lo_all) if lo_all else np.array([])
    hi_all = np.concatenate(hi_all) if hi_all else np.array([])
    act_all = np.concatenate(act_all) if act_all else np.array([])
    n = len(act_all)
    if n == 0:
        return {"n": 0}

    inside = (act_all >= lo_all) & (act_all <= hi_all)
    mean_width = float(np.mean(hi_all - lo_all))
    return {
        "n": int(n),
        "recorded_windows": int(recorded),
        "evaluated_windows": int(total),
        "evaluated_sequences": len(seq_windows),
        "k": int(k),
        "coverage": float(np.mean(inside)),
        "p10_coverage": float(np.mean(act_all >= lo_all)),
        "p90_coverage": float(np.mean(act_all <= hi_all)),
        "mean_width_kwh": mean_width,
        "median_width_kwh": float(np.median(hi_all - lo_all)),
        # PINAW per recorded window = interval width / own sequence's raw
        # load range (scale-free across heterogeneous meters), then averaged
        "pinaw": float(np.nanmean(rel_w_all)) if rel_w_all else float("nan"),
        "seq_range_kwh": {"min": float(min(seq_range.values())),
                          "median": float(np.median(list(seq_range.values()))),
                          "max": float(max(seq_range.values()))},
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def setup_cn_font() -> bool:
    import matplotlib
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                 "Source Han Sans SC", "WenQuanYi Zen Hei", "PingFang SC"):
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    print("WARNING: no CJK font found — falling back to English labels")
    return False


def make_figure(results: dict, eps_list: list[str], fig_path: Path,
                cn: bool) -> None:
    """Two panels: aggregate coverage + mean PINAW; per-client coverage bars
    (per-client PINAW lives in the standalone PINAW figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = TARGET_COVERAGE
    agg = results["aggregate"]
    per = results["per_client"]
    cids = results["client_order"]
    x = np.arange(len(eps_list))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.5]})

    # -------- (a) aggregate: coverage lines + mean PINAW (twin axis) --------
    cov = [agg[e].get("coverage_pooled", agg[e].get("coverage")) * 100
           for e in eps_list]
    cov_macro = [agg[e].get("coverage_macro") * 100 for e in eps_list]
    pinaw = [agg[e].get("pinaw_macro") * 100 for e in eps_list]

    ax1.axhline(T * 100, color="#2ca02c", ls="--", lw=1.4,
                label=(f"目标覆盖率 {T * 100:.0f}%" if cn
                       else f"target coverage {T * 100:.0f}%"))
    ax1.plot(x, cov, "-o", color="#08519c", lw=2.2, ms=7, zorder=3,
             label=("dp+best-RC 整体覆盖率（合并）" if cn
                    else "pooled coverage"))
    ax1.plot(x, cov_macro, "-^", color="#74c476", lw=1.4, ms=5, zorder=3,
             label=("客户端均值" if cn else "macro mean"))
    for xi, c in zip(x, cov):
        ax1.annotate(f"{c:.1f}", (xi, c), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=9, color="#08519c")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"ε={e}" for e in eps_list])
    ax1.set_ylabel("P10-P90 覆盖率 (%)" if cn else "P10-P90 coverage (%)")
    ax1.set_ylim(60, 100)
    ax1.grid(axis="y", color="#b0b0b0", lw=0.6, alpha=0.9)

    ax1b = ax1.twinx()
    ax1b.plot(x, pinaw, "-s", color="#e8912d", lw=2.0, ms=7, zorder=3,
              label=("平均 PINAW（右轴）" if cn else "mean PINAW (right)"))
    for xi, v in zip(x, pinaw):
        ax1b.annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                      xytext=(0, 9), ha="center", fontsize=9, color="#b06a10")
    ax1b.set_ylabel("PINAW (%)" if cn else "PINAW (%)", color="#b06a10")
    ax1b.tick_params(axis="y", colors="#b06a10")
    lo_p, hi_p = min(pinaw), max(pinaw)
    pad = max((hi_p - lo_p) * 0.35, 0.5)
    ax1b.set_ylim(max(0, lo_p - pad * 2.2), hi_p + pad * 1.6)

    # legend in the free upper-right zone (coverage curves sit at 60%-77%)
    ax1.legend(loc="upper right", fontsize=9)

    # -------- (b) per-client: coverage bars + PINAW scatter (twin axis) --------
    n_e, n_c = len(eps_list), len(cids)
    w = 0.8 / n_e
    xmax = 0.0
    for i, e in enumerate(eps_list):
        vals = [per[c][e]["coverage"] * 100 if per[c].get(e, {}).get("n")
                else np.nan for c in cids]
        xmax = max(xmax, np.nanmax(vals)
                   if not all(np.isnan(v) for v in vals) else 0)
        ax2.bar(np.arange(n_c) + (i - n_e / 2 + 0.5) * w, vals, w,
                label=f"ε={e}", color=EPS_COLORS[e], edgecolor="white", lw=0.4)
    ax2.axhline(T * 100, color="#2ca02c", ls="--", lw=1.4)
    ax2.set_xticks(np.arange(n_c))
    ax2.set_xticklabels(cids, rotation=30, ha="right", fontsize=8.5)
    ax2.set_ylabel("P10-P90 覆盖率 (%)" if cn else "P10-P90 coverage (%)")
    ax2.set_ylim(min(55, min(cov) - 6), max(xmax, T * 100) * 1.06)
    ax2.grid(axis="y", color="#b0b0b0", lw=0.6, alpha=0.9)
    ax2.legend(loc="upper right", ncol=3, fontsize=8)

    fig.suptitle(("dp+best-RC 区间覆盖率与 PINAW（P10-P90，五档隐私预算，最终轮模型）"
                  if cn else
                  "dp+best-RC interval coverage and PINAW across privacy budgets"),
                 fontsize=13)
    fig.text(0.99, 0.01,
             ("PINAW = 区间平均宽度 / 该客户端测试段真实负荷极差；统计在反归一化（原始 kWh）后进行"
              if cn else
              "PINAW = mean interval width / client test-region load range"),
             ha="right", va="bottom", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def make_pinaw_figure(results: dict, eps_list: list[str], fig_path: Path,
                      cn: bool) -> None:
    """Standalone figure: per-client PINAW, one coloured line per budget."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per = results["per_client"]
    cids = results["client_order"]
    x = np.arange(len(cids))
    line_colors = {  # five easily distinguishable colours (budget -> colour)
        "0.5": "#1f77b4", "1.5": "#d62728", "3.5": "#2ca02c",
        "5.5": "#9467bd", "7.5": "#ff7f0e",
    }

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    pin_max = 0.0
    for e in eps_list:
        vals = [(per[c].get(e) or {}).get("pinaw") for c in cids]
        vals = [v * 100 if v is not None and not np.isnan(v) else np.nan
                for v in vals]
        pin_max = max(pin_max, np.nanmax(vals)
                      if not all(np.isnan(v) for v in vals) else 0)
        ax.plot(x, vals, "-o", color=line_colors.get(e, "#555555"),
                lw=1.9, ms=5.5, markeredgecolor="white", markeredgewidth=0.5,
                label=f"ε={e}")
    ax.set_xticks(x)
    ax.set_xticklabels(cids, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("PINAW (%)")
    ax.set_ylim(0, pin_max * 1.15 if pin_max > 0 else 1)
    ax.grid(axis="y", color="#b0b0b0", lw=0.6, alpha=0.9)
    ax.legend(loc="upper right", fontsize=9,
              title=("隐私预算" if cn else "budget"), title_fontsize=9)

    fig.suptitle(("dp+best-RC 各客户端 PINAW（五档隐私预算，最终轮模型）"
                  if cn else
                  "dp+best-RC per-client PINAW across privacy budgets"),
                 fontsize=13)
    fig.text(0.99, 0.01,
             ("PINAW = 区间平均宽度 / 该客户端测试段真实负荷极差"
              if cn else
              "PINAW = mean interval width / client test-region load range"),
             ha="right", va="bottom", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    eps_list = [e.replace("epsilon-", "") for e in args.eps]

    client_ids = list_clients(args.clients)
    print(f"Clients ({len(client_ids)}): {', '.join(client_ids)}")

    per_client: dict[str, dict[str, dict]] = {cid: {} for cid in client_ids}
    best_rc_map: dict[str, dict[str, str]] = {}

    for eps in eps_list:
        ckpt = resolve_round_ckpt(eps, args.round)
        model = load_tcn(ckpt, device)
        print(f"\n=== ε={eps}  global={ckpt.name} ===")
        best_rc_map[eps] = {}

        for cid in client_ids:
            rc_type, rc_wape = best_rc_type(eps, cid)
            best_rc_map[eps][cid] = rc_type

            df, info = load_client_data(cid)
            feat_names = set(info["public_features"] + info["local_features"])
            seqs = [c for c in df.columns
                    if c not in feat_names and c != "datetime"]
            df_norm, params = preprocess(df, seqs, info["local_features"])

            ckpt_rc = (PERSONALIZED_DIR / f"epsilon-{eps}" / rc_type
                       / f"corrector_{cid}.pt")
            if not ckpt_rc.exists():
                print(f"  {cid}: corrector missing ({ckpt_rc}) — skipped")
                continue
            corrector = load_corrector(ckpt_rc, rc_type,
                                       len(info["local_features"]), device)

            res = rolling_eval_intervals(
                model, corrector, df_norm, seqs, info["public_features"],
                info["local_features"], params, args.max_windows, device,
                args.max_chain_windows)
            res["rc_type"] = rc_type
            res["rc_wape_selection"] = rc_wape
            per_client[cid][eps] = res
            if res.get("n"):
                print(f"  {cid:15s} rc={rc_type:4s} n={res['n']:6d} "
                      f"coverage={res['coverage'] * 100:5.1f}%  "
                      f"(P10≥:{res['p10_coverage'] * 100:5.1f}% "
                      f"P90≤:{res['p90_coverage'] * 100:5.1f}%) "
                      f"width={res['mean_width_kwh']:.1f} kWh")
            else:
                print(f"  {cid}: no valid windows")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # --- aggregates: pooled (primary) + macro ---
    aggregate: dict[str, dict] = {}
    for eps in eps_list:
        hits = n_tot = 0
        covs, pinaws = [], []
        for cid in client_ids:
            r = per_client[cid].get(eps)
            if r and r.get("n"):
                hits += int(round(r["coverage"] * r["n"]))
                n_tot += r["n"]
                covs.append(r["coverage"])
                if not math.isnan(r.get("pinaw", float("nan"))):
                    pinaws.append(r["pinaw"])
        aggregate[eps] = {
            "coverage_pooled": hits / n_tot if n_tot else float("nan"),
            "coverage_macro": float(np.mean(covs)) if covs else float("nan"),
            "mean_width_kwh": (float(np.mean([per_client[c][eps]["mean_width_kwh"]
                                              for c in client_ids
                                              if per_client[c].get(eps, {}).get("n")]))
                               if n_tot else float("nan")),
            "pinaw_macro": float(np.mean(pinaws)) if pinaws else float("nan"),
            "n_total": n_tot,
            "num_clients": len(covs),
            "target": TARGET_COVERAGE,
        }
        a = aggregate[eps]
        print(f"[ε={eps:4s}] pooled coverage={a['coverage_pooled'] * 100:.2f}%  "
              f"macro={a['coverage_macro'] * 100:.2f}%  "
              f"PINAW={a['pinaw_macro'] * 100:.2f}%  n={n_tot}")

    out_json = Path(args.output_json) if args.output_json else DEFAULT_JSON
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "script": "fl_code.coverage_eval",
            "args": {k: str(v) for k, v in vars(args).items()},
            "models": {e: str(resolve_round_ckpt(e, args.round)) for e in eps_list},
            "best_rc": best_rc_map,
            "note": ("P10-P90 coverage of dp+best-RC (per-client best corrector "
                     "by dp+rc WAPE) on the test portion, raw kWh units; final-"
                     "round DP global checkpoints; per-client recording capped "
                     "at --max-windows with the prev-residual chain intact"),
            "client_order": client_ids,
            "per_client": per_client,
            "aggregate": aggregate,
        }, f, indent=2, default=str)
    print(f"\nSaved JSON: {out_json}")

    cn = setup_cn_font()
    fig_dir = Path(args.fig_dir) if args.fig_dir else DEFAULT_FIG_DIR
    fig_path = fig_dir / "fig_coverage_epsilon.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    make_figure({"aggregate": aggregate, "per_client": per_client,
                 "client_order": client_ids}, eps_list, fig_path, cn)
    print(f"Saved: {fig_path}")
    pin_path = fig_dir / "fig_pinaw_per_client.png"
    make_pinaw_figure({"per_client": per_client, "client_order": client_ids},
                      eps_list, pin_path, cn)
    print(f"Saved: {pin_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P10-P90 interval coverage for dp+best-RC across the "
                    "epsilon sweep (final-round DP global checkpoints)")
    parser.add_argument("--eps", nargs="*", default=list(EPS_LIST),
                        help=f"Epsilon labels (default: {list(EPS_LIST)})")
    parser.add_argument("--round", type=int, default=30,
                        help="Checkpoint round to use (default: 30 = final)")
    parser.add_argument("--max-windows", type=int, default=3000,
                        help="Recorded windows per client (default: 3000)")
    parser.add_argument("--max-chain-windows", type=int, default=20000,
                        help="Chain-evaluated windows per client, sequences in "
                             "order (default: 20000)")
    parser.add_argument("--clients", nargs="*", default=None,
                        help="Client ids to include (default: all)")
    parser.add_argument("--output-json", type=str, default=None,
                        help=f"(default: {DEFAULT_JSON})")
    parser.add_argument("--fig-dir", type=str, default=None,
                        help=f"(default: {DEFAULT_FIG_DIR})")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda/cpu (default: auto)")
    args = parser.parse_args()
    main(args)
