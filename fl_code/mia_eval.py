"""Membership-inference attack (MIA) evaluation: no-DP vs DP epsilon sweep.

Attack setting
--------------
Black-box record-level membership inference on the **final-round Global TCN**
(the artefact that the FL server distributes to participants).  The attacker
scores a candidate load window by the model's prediction error — windows seen
during training tend to exhibit systematically lower error (memorisation) —
and predicts ``member`` when the score is low.

    score(window) = -mean_t | y_pre[t] - y_true[t] |     (normalised space)

Windows are sampled **non-overlapping** (stride = INPUT_STEPS + PRED_LEN =
150) from the chronological 80/20 split so that member and non-member window
pairs never share timesteps; a window is a *member* iff its label segment
lies entirely inside the training region.  For clients whose non-overlapping
test region yields fewer than ``--min-nonmembers`` windows, non-members are
sampled at a reduced stride (30) instead, and this is recorded in the JSON.

Drift control
-------------
Because the split is chronological, test-region windows are also further in
time and can be harder to predict *without any memorisation*.  To quantify
this confounder, the same member/non-member sets are additionally scored
with a **model-free seasonal-naive predictor** (y_hat[t] = load[t - 48],
24 h lag); its AUC is the "drift floor" attributable to distribution shift
alone.  AUC of the model attack should be compared against this floor.

Variants
--------
``nodp`` (``baseline_outputs/no-dp``) and the DP sweep
(``baseline_outputs/dp/epsilon-{0.5,1.5,3.5,5.5,7.5}``), all at
``--round`` (default 30, the final round).  Metrics per variant/client:

  - ``auc``                 attack ROC-AUC (0.5 = random guessing)
  - ``balanced_accuracy``   best (tpr + tnr) / 2 over thresholds
  - ``auc_excess``          auc - 0.5 (memorisation signal above chance)
  - ``n_members / n_nonmembers``

Outputs
-------
- ``fl_code/analysis/mia/mia_results.json``
- ``fl_code/analysis/figs/fig_mia_epsilon.png``
  (macro AUC vs epsilon + per-client bars, with the no-DP and
   drift-floor reference lines)

Usage::

    python -m fl_code.mia_eval
    python -m fl_code.mia_eval --round 30 --max-members 2000
    python -m fl_code.mia_eval --variants nodp dp --eps 1.5 7.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

from fl_code.config import INPUT_STEPS, PRED_LEN, TRAIN_RATIO
from fl_code.data_utils import load_client_data, preprocess, make_sliding_windows
from fl_code.models import TCNConfig, build_tcn

ROOT = Path(__file__).resolve().parents[1]
CLIENT_CONFIG_PATH = ROOT / "fl_code" / "models" / "client_config.yaml"
BASELINE_DIR = ROOT / "fl_code" / "baseline_outputs"
DEFAULT_JSON = ROOT / "fl_code" / "analysis" / "mia" / "mia_results.json"
DEFAULT_FIG_DIR = ROOT / "fl_code" / "analysis" / "figs"

EPS_LIST = ("0.5", "1.5", "3.5", "5.5", "7.5")
PAIR_STRIDE = INPUT_STEPS + PRED_LEN     # 150 → fully disjoint windows
LAG_24H = 48                             # seasonal-naive lag (30-min steps)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def resolve_round_ckpt(variant: str, eps: str | None, round_no: int) -> Path:
    if variant == "nodp":
        run_dir = BASELINE_DIR / "no-dp"
    else:
        run_dir = BASELINE_DIR / "dp" / f"epsilon-{eps}"
    exact = run_dir / "checkpoints" / f"round_{round_no:03d}.pt"
    if exact.exists():
        return exact
    files = sorted(run_dir.glob("checkpoints/round_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise FileNotFoundError(f"no checkpoints under {run_dir / 'checkpoints'}")
    return files[-1]


def load_tcn(path: Path, device: str) -> torch.nn.Module:
    model = build_tcn(TCNConfig()).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def list_clients(whitelist: list[str] | None = None) -> list[str]:
    with open(CLIENT_CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    ids = [cid for ds in config.values() for cid in ds["clients"]]
    return [c for c in ids if not whitelist or c in whitelist]


# ---------------------------------------------------------------------------
# Member / non-member window sets
# ---------------------------------------------------------------------------

def member_nonmember_windows(df_norm, seqs: list[str], public_cols: list[str],
                             max_members: int, min_nonmembers: int
                             ) -> tuple[dict, dict, dict]:
    """Build disjoint member / non-member window sets for one client.

    Returns ``{"X": ..., "y": ..., "meta": [...]}`` dicts with the same
    layout as :func:`data_utils.make_sliding_windows`, plus per-sequence
    split positions for the naive-drift scorer.
    """
    splits: dict[str, int] = {}
    for s in seqs:
        f = df_norm[s].first_valid_index()
        l = df_norm[s].last_valid_index()
        if f is None or l is None:
            continue
        splits[s] = f + int((l - f + 1) * TRAIN_RATIO)

    def collect(train: bool) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        # first pass at the pair-disjoint stride; widen non-member sampling
        # only if the client's test region is too small
        stride = PAIR_STRIDE
        for attempt in range(2):
            X, y, _, meta = make_sliding_windows(
                df_norm, seqs, public_cols, stride=stride, train=train)
            # drop boundary-straddling member labels (label must end <= split)
            if train:
                keep = [i for i, m in enumerate(meta)
                        if m["window_start"] + PAIR_STRIDE <= splits[m["seq"]]]
                X, y, meta = X[keep], y[keep], [meta[i] for i in keep]
            if train or len(meta) >= min_nonmembers or attempt == 1:
                return X, y, meta
            stride = 30                   # correlated but usable sample
        raise RuntimeError("unreachable")

    X_m, y_m, meta_m = collect(True)
    X_n, y_n, meta_n = collect(False)

    # evenly subsample members (keeps coverage across the training region)
    if len(meta_m) > max_members:
        idx = np.unique(np.linspace(0, len(meta_m) - 1, max_members).astype(int))
        X_m, y_m, meta_m = X_m[idx], y_m[idx], [meta_m[i] for i in idx]

    return ({"X": X_m, "y": y_m, "meta": meta_m},
            {"X": X_n, "y": y_n, "meta": meta_n},
            splits)


@torch.no_grad()
def model_scores(model: torch.nn.Module, sets: dict, device: str) -> np.ndarray:
    """Attack score = -normalised MAE, batched."""
    X = sets["X"]
    if len(X) == 0:
        return np.array([])
    out = []
    for c0 in range(0, len(X), 4096):
        xb = torch.from_numpy(X[c0:c0 + 4096]).to(device)
        pred = model(xb).cpu().numpy()
        out.append(-np.abs(pred - sets["y"][c0:c0 + 4096]).mean(axis=1))
    return np.concatenate(out)


def naive_scores(loads: dict[str, np.ndarray], sets: dict) -> np.ndarray:
    """Drift-floor score: seasonal-naive (24 h lag) prediction error."""
    out = []
    for m in sets["meta"]:
        s, i = m["seq"], m["window_start"]
        load = loads[s]
        seg_pred = load[i + INPUT_STEPS - LAG_24H: i + INPUT_STEPS + PRED_LEN - LAG_24H]
        seg_true = load[i + INPUT_STEPS: i + INPUT_STEPS + PRED_LEN]
        out.append(-np.abs(seg_pred - seg_true).mean())
    return np.array(out)


def attack_metrics(y_true: np.ndarray, score: np.ndarray) -> dict | None:
    """AUC + best balanced accuracy (score: higher = more member-like)."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    auc = float(roc_auc_score(y_true, score))
    fpr, tpr, thr = roc_curve(y_true, score)
    tnr = 1.0 - fpr
    bacc = float(np.max((tpr + tnr) / 2.0))
    return {"auc": auc, "balanced_accuracy": bacc,
            "auc_excess": auc - 0.5,
            "n_members": int((y_true == 1).sum()),
            "n_nonmembers": int((y_true == 0).sum())}


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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per = results["per_client"]
    agg = results["aggregate"]
    cids = results["client_order"]
    variants = ["nodp"] + [f"dp-{e}" for e in eps_list]
    x = np.arange(len(eps_list))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.5]})

    # (a) macro AUC vs epsilon
    auc = [agg[f"dp-{e}"]["auc_macro"] * 100 for e in eps_list]
    nodp = agg.get("nodp")
    nodp_auc = nodp["auc_macro"] * 100 if nodp else float("nan")

    ax1.axhline(50, color="#888888", ls=":", lw=1.4,
                label="随机猜测 50%" if cn else "random guessing")
    floors = [(per[c].get("nodp") or {}).get("auc_naive_drift_floor")
              for c in cids]
    floors = [f for f in floors if f is not None]
    drift = float(np.mean(floors)) * 100 if floors else None
    if drift is not None:
        ax1.axhline(drift, color="#e8912d", ls="-.", lw=1.4,
                    label=f"漂移本底 {drift:.1f}%" if cn else f"drift floor")
    if nodp:
        ax1.axhline(nodp_auc, color="#d62728", ls="--", lw=1.6,
                    label=f"no-DP {nodp_auc:.1f}%")
    ax1.plot(x, auc, "-o", color="#08519c", lw=2.2, ms=7, zorder=3,
             label="DP 全局模型" if cn else "DP global model")
    for xi, a in zip(x, auc):
        ax1.annotate(f"{a:.1f}", (xi, a), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9, color="#08519c")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"ε={e}" for e in eps_list])
    ax1.set_ylabel("成员推断攻击 AUC (%)" if cn else "MIA ROC-AUC (%)")
    vals = list(auc) + [50]
    if nodp:
        vals.append(nodp_auc)
    if drift is not None:
        vals.append(drift)
    ax1.set_ylim(min(vals) - 3, max(vals) + 4)
    ax1.grid(axis="y", color="#b0b0b0", lw=0.6, alpha=0.9)
    ax1.legend(loc="lower left", fontsize=8.5)

    # (b) per-client grouped bars: nodp vs each epsilon
    bar_colors = {"nodp": "#d62728"}
    blues = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c"]
    for i, e in enumerate(eps_list):
        bar_colors[f"dp-{e}"] = blues[i]
    n_v, n_c = len(variants), len(cids)
    w = 0.8 / n_v
    for i, v in enumerate(variants):
        vals = [(per[c].get(v) or {}).get("auc")
                for c in cids]
        vals = [v_ * 100 if v_ is not None else np.nan for v_ in vals]
        ax2.bar(np.arange(n_c) + (i - n_v / 2 + 0.5) * w, vals, w,
                label=("no-DP" if v == "nodp" else f"ε={v.split('-')[1]}"),
                color=bar_colors[v], edgecolor="white", lw=0.3)
    ax2.axhline(50, color="#888888", ls=":", lw=1.2)
    ax2.set_xticks(np.arange(n_c))
    ax2.set_xticklabels(cids, rotation=30, ha="right", fontsize=8.5)
    ax2.set_ylabel("成员推断攻击 AUC (%)" if cn else "MIA ROC-AUC (%)")
    ax2.set_ylim(40, 100)
    ax2.grid(axis="y", color="#b0b0b0", lw=0.6, alpha=0.9)
    ax2.legend(loc="upper right", ncol=3, fontsize=8)

    fig.suptitle(("成员推断攻击：无DP vs 差分隐私（各档最终轮全局模型）"
                  if cn else
                  "Membership inference: no-DP vs DP (final-round checkpoints)"),
                 fontsize=13)
    fig.text(0.99, 0.01,
             "黑盒攻击：窗口级误差打分，成员/非成员窗口互不重叠；AUC=50% 表示无法区分",
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
    eps_list = [e.replace("dp-", "").replace("epsilon-", "") for e in args.eps]
    variants: list[tuple[str, str | None]] = [("nodp", None)] + \
        [("dp-" + e, e) for e in eps_list]
    if not args.nodp:
        variants = variants[1:]

    client_ids = list_clients(args.clients)
    print(f"Clients ({len(client_ids)}): {', '.join(client_ids)}")

    # window sets are shared across variants (same split, same sampling)
    client_sets: dict[str, tuple[dict, dict, dict, object]] = {}
    for cid in client_ids:
        df, info = load_client_data(cid)
        feat_names = set(info["public_features"] + info["local_features"])
        seqs = [c for c in df.columns if c not in feat_names and c != "datetime"]
        df_norm, _ = preprocess(df, seqs, info["local_features"])
        mem, non, splits = member_nonmember_windows(
            df_norm, seqs, info["public_features"],
            args.max_members, args.min_nonmembers)
        loads = {s: df_norm[s].values.astype(np.float32) for s in seqs}
        naive_m = naive_scores(loads, mem) if len(mem["meta"]) else np.array([])
        naive_n = naive_scores(loads, non) if len(non["meta"]) else np.array([])
        client_sets[cid] = (mem, non, (naive_m, naive_n), df_norm)
        print(f"{cid:15s} members={len(mem['meta']):5d} "
              f"nonmembers={len(non['meta']):5d}")

    per_client: dict[str, dict[str, dict]] = {cid: {} for cid in client_ids}

    for vname, eps in variants:
        ckpt = resolve_round_ckpt("nodp" if vname == "nodp" else "dp", eps,
                                  args.round)
        model = load_tcn(ckpt, device)
        print(f"\n=== {vname}  global={ckpt.name} ===")

        aucs, baccs = [], []
        for cid in client_ids:
            mem, non, (naive_m, naive_n), _ = client_sets[cid]
            y_true = np.concatenate([np.ones(len(mem["meta"])),
                                     np.zeros(len(non["meta"]))])
            s_model = np.concatenate([model_scores(model, mem, device),
                                      model_scores(model, non, device)])
            met = attack_metrics(y_true, s_model)
            if met is None:
                print(f"  {cid:15s} skipped (single class)")
                continue
            s_naive = np.concatenate([naive_m, naive_n])
            met_naive = attack_metrics(y_true, s_naive)
            met["auc_naive_drift_floor"] = (met_naive or {}).get("auc")
            per_client[cid][vname] = met
            aucs.append(met["auc"])
            baccs.append(met["balanced_accuracy"])
            print(f"  {cid:15s} AUC={met['auc'] * 100:5.1f}%  "
                  f"bAcc={met['balanced_accuracy'] * 100:5.1f}%  "
                  f"(n={met['n_members']}/{met['n_nonmembers']}"
                  + (f", drift floor {met['auc_naive_drift_floor'] * 100:.1f}%)"
                     if met_naive else ")"))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        per_client.setdefault("_aggregate", {})[vname] = {
            "auc_macro": float(np.mean(aucs)) if aucs else float("nan"),
            "balanced_accuracy_macro": (float(np.mean(baccs)) if baccs
                                        else float("nan")),
            "num_clients": len(aucs),
        }
        a = per_client["_aggregate"][vname]
        print(f"  [{vname:8s}] macro AUC={a['auc_macro'] * 100:.2f}%  "
              f"macro bAcc={a['balanced_accuracy_macro'] * 100:.2f}%")

    # --- relative drop (榜题口径: 攻击指标相对下降) ---
    aggregate: dict[str, dict] = dict(per_client.pop("_aggregate"))
    if "nodp" in aggregate:
        floors = [(per_client[c].get("nodp") or {}).get("auc_naive_drift_floor")
                  for c in client_ids]
        floors = [x for x in floors if x is not None]
        if floors:
            aggregate["nodp"]["auc_macro_naive_drift_floor"] = float(np.mean(floors))
    rel = {}
    if "nodp" in aggregate:
        base_auc = aggregate["nodp"]["auc_macro"]
        base_acc = aggregate["nodp"]["balanced_accuracy_macro"]
        for vname, _ in variants:
            if vname == "nodp":
                continue
            a = aggregate[vname]
            rel[vname] = {
                "auc_relative_drop_vs_nodp": (base_auc - a["auc_macro"]) / base_auc,
                "bacc_relative_drop_vs_nodp":
                    (base_acc - a["balanced_accuracy_macro"]) / base_acc,
            }
            r = rel[vname]
            print(f"[{vname:8s}] AUC relative drop vs nodp = "
                  f"{r['auc_relative_drop_vs_nodp'] * 100:.1f}%  "
                  f"bAcc relative drop = {r['bacc_relative_drop_vs_nodp'] * 100:.1f}%")

    out_json = Path(args.output_json) if args.output_json else DEFAULT_JSON
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "script": "fl_code.mia_eval",
            "args": {k: str(v) for k, v in vars(args).items()},
            "models": {v: str(resolve_round_ckpt("nodp" if v == "nodp" else "dp",
                                                 e, args.round))
                       for v, e in variants},
            "note": ("black-box record-level MIA on the final-round global TCN; "
                     "score = -normalised window MAE; member/non-member windows "
                     "non-overlapping (stride 150; non-members at stride 30 for "
                     "small test regions); 'auc_naive_drift_floor' = same attack "
                     "with a model-free 24h seasonal-naive scorer, quantifying "
                     "the chronological-drift confounder"),
            "relative_drop_vs_nodp": rel,
            "client_order": client_ids,
            "per_client": per_client,
            "aggregate": aggregate,
        }, f, indent=2, default=str)
    print(f"\nSaved JSON: {out_json}")

    cn = setup_cn_font()
    fig_dir = Path(args.fig_dir) if args.fig_dir else DEFAULT_FIG_DIR
    fig_path = fig_dir / "fig_mia_epsilon.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    make_figure({"per_client": per_client, "aggregate": aggregate,
                 "client_order": client_ids}, eps_list, fig_path, cn)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Membership-inference attack comparison: no-DP vs DP "
                    "epsilon sweep (final-round global checkpoints)")
    parser.add_argument("--eps", nargs="*", default=list(EPS_LIST),
                        help=f"Epsilon labels (default: {list(EPS_LIST)})")
    parser.add_argument("--round", type=int, default=30,
                        help="Checkpoint round (default: 30 = final)")
    parser.add_argument("--max-members", type=int, default=2000,
                        help="Member windows per client (default: 2000)")
    parser.add_argument("--min-nonmembers", type=int, default=50,
                        help="If fewer non-overlapping non-members, fall back "
                             "to stride-30 sampling (default: 50)")
    parser.add_argument("--no-nodp", dest="nodp", action="store_false",
                        help="Skip the no-DP variant")
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
