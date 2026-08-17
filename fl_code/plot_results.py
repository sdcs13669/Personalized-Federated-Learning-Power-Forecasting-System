"""Plot training / validation results from Phase 2 & 3 output JSONs.

Reads:
  - ``fl_code/baseline_outputs/nodp/baseline_history.json`` (Phase 1: FedAvg, no DP)
  - ``fl_code/baseline_outputs/dp/baseline_history.json`` (Phase 2: DP-FedAvg)
  - ``fl_code/personalized_outputs/<rc_type>/personalized_results.json``
    (Phase 3: dp+rc; rc type chosen via ``--rc-type``)

Produces one figure per stage (``fl_code/figures/``), each with the
training-loss curves on the left and the validation-metric bars on the right,
plus a 3-stage comparison figure:

  - ``fig_phase1_nodp.png``           left: per-client per-round training loss
                                      right: MAE / RMSE / R² grouped bars
  - ``fig_phase2_dp.png``             same layout for the DP variant
  - ``fig_phase3_personalized.png``   left: per-client per-epoch corrector loss
                                      right: Y_pre vs Y_final MAE bars
  - ``fig_phase_compare.png``         left: WAPE / avg MAE per stage
                                      right: per-client MAE / RMSE per stage

All metrics are computed on normalised data (before de-normalisation) —
each figure carries a note saying so.

Usage::

    python -m fl_code.plot_results                        # save PNGs (rc type: mlp)
    python -m fl_code.plot_results --rc-type tcn          # pick a Phase 3 rc type
    python -m fl_code.plot_results --personalized-json my/path/personalized_results.json
    python -m fl_code.plot_results --show                 # also open figure windows
    python -m fl_code.plot_results --root my_run          # custom baseline output root
    python -m fl_code.plot_results --fig-dir my_figs      # custom figure output dir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "fl_code" / "figures"
_LEGACY_PERSONALIZED_JSON = (ROOT / "fl_code" / "personalized_outputs"
                             / "personalized_results.json")


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
            "wape": "WAPE",
            "avg_mae": "平均 MAE",
            "fig_p1": "阶段一 基线 (FedAvg, 无DP)",
            "fig_p2": "阶段二 基线 (DP-FedAvg)",
            "fig_p3": "阶段三 个性化",
            "fig_compare": "三阶段性能对比 (nodp / dp / dp+rc)",
            "cmp_left": "各阶段 WAPE / 平均 MAE",
            "cmp_right": "各阶段每客户端 MAE / RMSE",
            "wape_na": "dp+rc 的 WAPE 需重跑阶段三后写入",
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
        "wape": "WAPE",
        "avg_mae": "avg MAE",
        "fig_p1": "Phase 1 Baseline (FedAvg, no DP)",
        "fig_p2": "Phase 2 Baseline (DP-FedAvg)",
        "fig_p3": "Phase 3 Personalised",
        "fig_compare": "3-Stage Comparison (nodp / dp / dp+rc)",
        "cmp_left": "WAPE / avg MAE per stage",
        "cmp_right": "Per-client MAE / RMSE per stage",
        "wape_na": "WAPE for dp+rc requires re-running Phase 3",
    }


def _finish(fig, L: dict):
    """Shared suptitle/note/tight layout for a phase figure."""
    fig.text(0.99, 0.01, L["note"], ha="right", va="bottom",
             fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# ---------------------------------------------------------------------------
# Phase 2 figure — left: loss curves, right: validation bars
# ---------------------------------------------------------------------------

def _fig_phase2(baseline: dict, L: dict, plt, fig_title: str | None = None):
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

    fig.suptitle(fig_title or L["fig_p2"], fontsize=13)
    _finish(fig, L)
    return fig


# ---------------------------------------------------------------------------
# Phase 3 figure — left: loss curves, right: validation bars
# ---------------------------------------------------------------------------

def _fig_phase3(personalized: dict, L: dict, plt, rc_type: str = "mlp"):
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

    fig.suptitle(f"{L['fig_p3']} ({rc_type})", fontsize=13)
    _finish(fig, L)
    return fig


# ---------------------------------------------------------------------------
# 3-stage comparison figure
# ---------------------------------------------------------------------------

STAGE_COLORS = {"nodp": "#d62728", "dp": "#1f77b4", "dp+rc": "#2ca02c"}


def _stage_table(nodp: dict | None, dp: dict | None,
                 personalized: dict | None) -> dict[str, dict]:
    """Extract per-stage summary values for the comparison figure.

    Returns ``{"nodp": ..., "dp": ..., "dp+rc": ...}``; each entry is
    ``{"wape", "avg_mae", "clients": {cid: {"mae", "rmse"}}}`` — missing
    stages or keys become NaN / empty client dicts.
    """
    def stage(wape, avg_mae, client_metrics):
        clients = {
            cid: {"mae": float((m or {}).get("mae", np.nan)),
                  "rmse": float((m or {}).get("rmse", np.nan))}
            for cid, m in (client_metrics or {}).items()
        }
        return {"wape": float(wape), "avg_mae": float(avg_mae),
                "clients": clients}

    table: dict[str, dict] = {}
    for label, base in (("nodp", nodp), ("dp", dp)):
        if base is None:
            table[label] = {"wape": np.nan, "avg_mae": np.nan, "clients": {}}
            continue
        fm = base.get("final_metrics") or {}
        table[label] = stage(fm.get("wape", np.nan),
                             fm.get("avg_mae", np.nan),
                             fm.get("client_metrics"))
    if personalized is None:
        table["dp+rc"] = {"wape": np.nan, "avg_mae": np.nan, "clients": {}}
    else:
        fm = personalized.get("final_metrics") or {}
        table["dp+rc"] = stage(fm.get("wape_personalized", np.nan),
                               fm.get("avg_mae_personalized", np.nan),
                               fm.get("client_metrics"))
    return table


def _fig_compare(table: dict, L: dict, plt):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(17, 6))
    stages = ["nodp", "dp", "dp+rc"]

    # --- left: WAPE / avg MAE per stage ---
    wapes = [table[s]["wape"] for s in stages]
    avgs = [table[s]["avg_mae"] for s in stages]
    x = np.arange(len(stages))
    w = 0.35
    ax_left.bar(x - w / 2, [0 if np.isnan(v) else v for v in wapes], w,
                label=L["wape"], color="#d62728")
    ax_left.bar(x + w / 2, [0 if np.isnan(v) else v for v in avgs], w,
                label=L["avg_mae"], color="#2ca02c")
    if np.isnan(wapes[2]):
        ax_left.text(2 - w / 2, 0.01, L["wape_na"], ha="center", va="bottom",
                     fontsize=7, color="#555555")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(stages)
    ax_left.set_title(L["cmp_left"])
    ax_left.legend(fontsize=8)
    ax_left.grid(alpha=0.3, axis="y")

    # --- right: per-client MAE / RMSE per stage (6 bars per client) ---
    cids = sorted({cid for s in stages for cid in table[s]["clients"]})
    x = np.arange(len(cids))
    bw = 0.12
    offsets = (np.arange(6) - 2.5) * bw
    for s_i, s in enumerate(stages):
        for m_i, metric in enumerate(("mae", "rmse")):
            vals = [table[s]["clients"].get(c, {}).get(metric, np.nan)
                    for c in cids]
            ax_right.bar(x + offsets[s_i * 2 + m_i],
                         [0 if np.isnan(v) else v for v in vals], bw,
                         label=f"{s} {metric.upper()}",
                         color=STAGE_COLORS[s],
                         hatch="//" if metric == "rmse" else None)
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(cids, rotation=30, ha="right", fontsize=8)
    ax_right.set_title(L["cmp_right"])
    ax_right.legend(fontsize=7, ncol=2)
    ax_right.grid(alpha=0.3, axis="y")

    fig.suptitle(L["fig_compare"], fontsize=13)
    _finish(fig, L)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _personalized_json_path(rc_type: str) -> Path:
    return ROOT / "fl_code" / "personalized_outputs" / rc_type / "personalized_results.json"


def main(show: bool = False, root: Path | None = None, fig_dir: Path | None = None,
         rc_type: str = "mlp", personalized_json: Path | None = None):
    import matplotlib
    matplotlib.use("TkAgg" if show else "Agg")
    import matplotlib.pyplot as plt

    cn = _setup_cn_font()
    L = _labels(cn)

    root = root or (ROOT / "fl_code" / "baseline_outputs")
    baselines: dict[str, dict] = {}
    for variant in ("nodp", "dp"):
        cand = root / variant / "baseline_history.json"
        baseline = _load_json(cand)
        if baseline is not None:
            baselines[variant] = baseline
            print(f"Using {variant} results: {cand}")

    if personalized_json is None:
        personalized_json = _personalized_json_path(rc_type)
        if not personalized_json.exists() and _LEGACY_PERSONALIZED_JSON.exists():
            print(f"WARNING: {personalized_json} not found — "
                  f"using legacy {_LEGACY_PERSONALIZED_JSON}")
            personalized_json = _LEGACY_PERSONALIZED_JSON
    personalized = _load_json(personalized_json)
    if personalized is not None:
        # 仅当确实按 --rc-type 解析时才标注 rc 类型（显式 --personalized-json 不标）
        tag = (f" ({rc_type})"
               if personalized_json == _personalized_json_path(rc_type) else "")
        print(f"Using personalized results{tag}: {personalized_json}")
    if not baselines and personalized is None:
        raise SystemExit(
            f"No result files found:\n  {root / 'nodp' / 'baseline_history.json'}\n"
            f"  {root / 'dp' / 'baseline_history.json'}\n  {personalized_json}\n"
            f"Run train_baseline.py / train_personalized.py first.")

    out_dir = fig_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    if "nodp" in baselines:
        fig = _fig_phase2(baselines["nodp"], L, plt, fig_title=L["fig_p1"])
        p = out_dir / "fig_phase1_nodp.png"
        fig.savefig(p, dpi=150)
        saved.append(p)

    if "dp" in baselines:
        fig = _fig_phase2(baselines["dp"], L, plt, fig_title=L["fig_p2"])
        p = out_dir / "fig_phase2_dp.png"
        fig.savefig(p, dpi=150)
        saved.append(p)

    if personalized is not None:
        fig = _fig_phase3(personalized, L, plt, rc_type)
        p = out_dir / "fig_phase3_personalized.png"
        fig.savefig(p, dpi=150)
        saved.append(p)

    n_stages = len(baselines) + (1 if personalized is not None else 0)
    if n_stages >= 2:
        table = _stage_table(baselines.get("nodp"), baselines.get("dp"),
                             personalized)
        fig = _fig_compare(table, L, plt)
        p = out_dir / "fig_phase_compare.png"
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
                        help="Baseline output root; reads <root>/nodp and "
                             "<root>/dp baseline_history.json, whichever exist "
                             "(default: fl_code/baseline_outputs)")
    parser.add_argument("--fig-dir", type=str,
                        default=str(OUT_DIR),
                        help="Figure output directory "
                             "(default: fl_code/figures)")
    parser.add_argument("--rc-type", type=str, default="mlp",
                        choices=["mlp", "lstm", "tcn"],
                        help="Residual Corrector type for the Phase 3 results "
                             "(reads fl_code/personalized_outputs/<rc-type>/"
                             "personalized_results.json; default: mlp)")
    parser.add_argument("--personalized-json", type=str, default=None,
                        help="Explicit path to personalized_results.json "
                             "(overrides --rc-type resolution)")
    args = parser.parse_args()
    main(show=args.show, root=Path(args.root),
         fig_dir=Path(args.fig_dir), rc_type=args.rc_type,
         personalized_json=(Path(args.personalized_json)
                            if args.personalized_json else None))
