"""Plot training / validation results from Phase 2 & 3 output JSONs.

Reads:
  - ``fl_code/baseline_outputs/baseline_history.json``     (Phase 2 FedAvg)
  - ``fl_code/personalized_outputs/personalized_results.json`` (Phase 3)

Produces one figure per phase (``fl_code/figures/``), each with the
training-loss curves on the left and the validation-metric bars on the right:

  - ``fig_phase2_baseline.png``      left: per-client per-round training loss
                                     right: MAE / RMSE / R² grouped bars
  - ``fig_phase3_personalized.png``  left: per-client per-epoch corrector loss
                                     right: Y_pre vs Y_final MAE bars

All metrics are computed on normalised data (before de-normalisation) —
each figure carries a note saying so.

Usage::

    python -m fl_code.plot_results          # save PNGs
    python -m fl_code.plot_results --show   # also open figure windows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PERSONALIZED_JSON = ROOT / "fl_code" / "personalized_outputs" / "personalized_results.json"
OUT_DIR = ROOT / "fl_code" / "figures"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
            "note": "注：所有指标均在反归一化前（归一化空间）计算",
            "round": "通信轮次 (round)",
            "epoch": "本地训练轮次 (epoch)",
            "train_mae": "训练 MAE",
            "pinball": "Pinball 损失",
            "avg": "服务器平均",
            "p2_loss_title": "每客户端逐轮训练损失 (FedAvg)",
            "p2_val_title": "每客户端最终验证指标 (MAE / RMSE / R$^2$)",
            "p3_loss_title": "每客户端逐 epoch 训练损失 (Corrector)",
            "p3_val_title": "每客户端最终验证 MAE (Y_pre vs Y_final)",
            "mae": "MAE",
            "rmse": "RMSE",
            "r2": "R$^2$",
            "ypre": "Y_pre MAE (全局基线)",
            "yfinal": "Y_final MAE (个性化)",
            "fig_p2": "Phase 2 Baseline (FedAvg)",
            "fig_p3": "Phase 3 Personalised",
        }
    return {
        "note": "Note: all metrics computed before de-normalisation (normalised space)",
        "round": "Communication round",
        "epoch": "Local training epoch",
        "train_mae": "Training MAE",
        "pinball": "Pinball loss",
        "avg": "Server average",
        "p2_loss_title": "Per-client per-round training loss (FedAvg)",
        "p2_val_title": "Per-client final validation metrics (MAE / RMSE / R$^2$)",
        "p3_loss_title": "Per-client per-epoch training loss (Corrector)",
        "p3_val_title": "Per-client final validation MAE (Y_pre vs Y_final)",
        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R$^2$",
        "ypre": "Y_pre MAE (global baseline)",
        "yfinal": "Y_final MAE (personalised)",
        "fig_p2": "Phase 2 Baseline (FedAvg)",
        "fig_p3": "Phase 3 Personalised",
    }


def _finish(fig, L: dict):
    """Shared suptitle/note/tight layout for a phase figure."""
    fig.text(0.99, 0.01, L["note"], ha="right", va="bottom",
             fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# ---------------------------------------------------------------------------
# Phase 2 figure — left: loss curves, right: validation bars
# ---------------------------------------------------------------------------

def _fig_phase2(baseline: dict, L: dict, plt):
    fig, (ax_line, ax_bar) = plt.subplots(1, 2, figsize=(13.5, 5))

    # --- left: per-client per-round training loss ---
    per_client = baseline.get("train_losses_per_client") or {}
    series: dict[str, dict[int, float]] = {}
    for r_str, d in per_client.items():
        for cid, v in d.items():
            series.setdefault(cid, {})[int(r_str)] = v
    for cid, rd in series.items():
        rs = sorted(rd)
        ax_line.plot(rs, [rd[r] for r in rs], marker="o", ms=3, lw=1.2,
                     label=cid)
    avg = baseline.get("train_losses")
    if avg:
        ax_line.plot(np.arange(1, len(avg) + 1), avg, "k--", lw=1.5,
                     label=L["avg"])
    ax_line.set_xlabel(L["round"])
    ax_line.set_ylabel(L["train_mae"])
    ax_line.set_title(L["p2_loss_title"])
    ax_line.legend(fontsize=7)
    ax_line.grid(alpha=0.3)

    # --- right: MAE / RMSE / R² grouped bars ---
    cm = baseline.get("final_metrics", {}).get("client_metrics") or {}
    cids = list(cm)
    x = np.arange(len(cids))
    w = 0.27
    ax_bar.bar(x - w, [cm[c]["mae"] for c in cids], w, label=L["mae"],
               color="#d62728")                                   # red
    ax_bar.bar(x, [cm[c]["rmse"] for c in cids], w, label=L["rmse"],
               color="#2ca02c")                                   # green
    ax_bar.bar(x + w, [cm[c]["r2"] for c in cids], w, label=L["r2"],
               color="#1f77b4")                                   # blue
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(cids, rotation=30, ha="right", fontsize=8)
    ax_bar.set_title(L["p2_val_title"])
    ax_bar.legend(fontsize=8)
    ax_bar.grid(alpha=0.3, axis="y")

    fig.suptitle(L["fig_p2"], fontsize=13)
    _finish(fig, L)
    return fig


# ---------------------------------------------------------------------------
# Phase 3 figure — left: loss curves, right: validation bars
# ---------------------------------------------------------------------------

def _fig_phase3(personalized: dict, L: dict, plt):
    fig, (ax_line, ax_bar) = plt.subplots(1, 2, figsize=(13.5, 5))

    # --- left: per-client per-epoch corrector training loss ---
    for cid, r in personalized.get("results", {}).items():
        losses = r.get("epoch_losses") or []
        if losses:
            ax_line.plot(np.arange(1, len(losses) + 1), losses,
                         marker="o", ms=3, lw=1.2, label=cid)
    ax_line.set_xlabel(L["epoch"])
    ax_line.set_ylabel(L["pinball"])
    ax_line.set_title(L["p3_loss_title"])
    ax_line.legend(fontsize=7)
    ax_line.grid(alpha=0.3)

    # --- right: Y_pre vs Y_final MAE bars ---
    results = personalized.get("results", {})
    cids = list(results)
    x = np.arange(len(cids))
    w = 0.35
    ax_bar.bar(x - w / 2, [results[c]["mae_baseline"] for c in cids], w,
               label=L["ypre"], color="#d62728")
    ax_bar.bar(x + w / 2, [results[c]["mae_personalized"] for c in cids], w,
               label=L["yfinal"], color="#2ca02c")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(cids, rotation=30, ha="right", fontsize=8)
    ax_bar.set_title(L["p3_val_title"])
    ax_bar.legend(fontsize=8)
    ax_bar.grid(alpha=0.3, axis="y")

    fig.suptitle(L["fig_p3"], fontsize=13)
    _finish(fig, L)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(show: bool = False, root: Path | None = None):
    import matplotlib
    matplotlib.use("TkAgg" if show else "Agg")
    import matplotlib.pyplot as plt

    cn = _setup_cn_font()
    L = _labels(cn)

    root = root or (ROOT / "fl_code" / "baseline_outputs")
    baseline = None
    for cand in (root / "nodp" / "baseline_history.json",
                 root / "dp" / "baseline_history.json"):
        baseline = _load_json(cand)
        if baseline is not None:
            print(f"Using Phase 2 results: {cand}")
            break
    personalized = _load_json(PERSONALIZED_JSON)
    if baseline is None and personalized is None:
        raise SystemExit(
            f"No result files found:\n  {root / 'nodp' / 'baseline_history.json'}\n"
            f"  {root / 'dp' / 'baseline_history.json'}\n  {PERSONALIZED_JSON}\n"
            f"Run train_baseline.py / train_personalized.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    if baseline is not None:
        fig = _fig_phase2(baseline, L, plt)
        p = OUT_DIR / "fig_phase2_baseline.png"
        fig.savefig(p, dpi=150)
        saved.append(p)

    if personalized is not None:
        fig = _fig_phase3(personalized, L, plt)
        p = OUT_DIR / "fig_phase3_personalized.png"
        fig.savefig(p, dpi=150)
        saved.append(p)

    for p in saved:
        print(f"Saved: {p}")
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Phase 2/3 training & validation results")
    parser.add_argument("--show", action="store_true",
                        help="Open figure windows (GUI)")
    parser.add_argument("--root", type=str,
                        default=str(ROOT / "fl_code" / "baseline_outputs"),
                        help="Baseline output root; reads <root>/nodp or falls "
                             "back to <root>/dp baseline_history.json "
                             "(default: fl_code/baseline_outputs)")
    args = parser.parse_args()
    main(show=args.show, root=Path(args.root))
