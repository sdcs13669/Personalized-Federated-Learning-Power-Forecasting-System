"""Task 2: standard figure set (aligned with the DP-FL handoff spec).

Generates:
  Fig 1  privacy–utility trade-off     -> analysis/figs/fig_privacy_utility.png
  Fig 2  mode ablation (DP / RC gain)  -> analysis/figs/fig_rc_ablation.png
  Fig 3  per-client detail (one per ε) -> analysis/epsilon-<e>/fig_per_client_epsilon_<e>.png
  Fig 4  training convergence          -> analysis/figs/fig_convergence.png
  Fig 5  privacy audit (3 panels)      -> analysis/figs/fig_privacy_audit.png
  Table 6 summary table (markdown)     -> analysis/figs/summary_table.md
  Fig 7  RC architecture ablation      -> analysis/figs/fig_rc_arch_ablation.png

All labels are English (avoids CJK-font tofu on servers without Chinese fonts;
handoff spec explicitly permits English when Chinese fonts are unavailable).

Read-only inputs:
  - fl_code/analysis/epsilon-*/denorm_metrics_{mlp,lstm,tcn}.json
  - fl_code/baseline_outputs/no-dp/, dp/epsilon-*/ (baseline_history.json,
    audit_log.json, config.json)

Usage:
  python -m fl_code.make_figures                      # official output dirs
  python -m fl_code.make_figures --fig-root /tmp/x    # preview into /tmp/x (never touches official dirs)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ANALYSIS = ROOT / "fl_code" / "analysis"      # read-only metric inputs
BASELINE = ROOT / "fl_code" / "baseline_outputs"   # read-only training outputs
FIGS = DATA_ANALYSIS / "figs"                      # output: cross-ε figures
EPS_FIG_ROOT = DATA_ANALYSIS                       # output: per-ε figures
FIGS.mkdir(parents=True, exist_ok=True)

EPS = ["0.5", "1.5", "3.5", "5.5", "7.5"]
ARCHS = ["mlp", "lstm", "tcn"]
CLIENT_ORDER = ["steel_ind_0", "tetouan_city_0", "tetouan_city_1",
                "tetouan_city_2", "lcl_res_0", "lcl_res_1",
                "eld_ind_0", "eld_ind_1", "eld_ind_2"]
DATASETS = [("steel_ind", 1), ("tetouan_city", 3), ("lcl_res", 2),
            ("eld_ind", 3)]
DS_LABEL = {"steel_ind": "Steel industrial", "tetouan_city": "Urban grid",
            "lcl_res": "Residential", "eld_ind": "Industrial"}

# Semantic colors: nodp = red (baseline), dp = blue (privacy), dp+rc = green (recovered)
C = {"nodp": "#C44E52", "dp": "#4C72B0", "dp+rc": "#55A868"}
# Architecture ablation: neutral grays, best reuses dp+rc green
ARCH_C = {"mlp": "#a8b3bf", "lstm": "#66788a", "tcn": "#c9d1d9",
          "best": C["dp+rc"]}
# Privacy scale for dp convergence/audit lines: darker = larger ε
# (higher privacy budget converges closer to the non-DP baseline)
EPS_COLORS = {"0.5": "#c3d6e8", "1.5": "#6baed6", "3.5": "#4292c6",
              "5.5": "#2171b5", "7.5": "#084594"}
# Distinct categorical colors for overlaid per-ε curves (Fig 5C)
EPS_DISTINCT = {"0.5": "#4C72B0", "1.5": "#DD8452", "3.5": "#55A868",
                "5.5": "#C44E52", "7.5": "#8172B3"}

FOOT = ("De-normalised (raw kWh); pooled over 9 clients; "
        "dp+rc = per-client best RC")


def setup_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "grid.color": "#d6d6d6",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "svg.fonttype": "none",
    })


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"   saved {path.name}")


def footnote(fig, text: str):
    fig.text(0.99, 0.004, text, ha="right", va="bottom",
             fontsize=6, color="#666666")



# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_metrics():
    """Return {epsilon: {arch: data}}."""
    result = {}
    for e in EPS:
        result[e] = {}
        for a in ARCHS:
            p = DATA_ANALYSIS / f"epsilon-{e}" / f"denorm_metrics_{a}.json"
            if p.exists():
                with open(p) as f:
                    result[e][a] = json.load(f)
    return result


def _load_json(rel: Path):
    with open(rel) as f:
        return json.load(f)


def compute_best_rc(metrics, e: str, per_client: bool = False):
    """Per-client best RC (min dp+rc WAPE); aggregate = pooled over best picks."""
    m = metrics[e]["mlp"]["per_client"]
    if per_client:
        best = {}
        for cid in CLIENT_ORDER:
            if cid not in m:
                continue
            best_arch, best_wape = None, float("inf")
            for a in ARCHS:
                rc = metrics[e][a]["per_client"][cid].get("dp+rc")
                if rc and rc["wape"] < best_wape:
                    best_wape, best_arch = rc["wape"], a
            best[cid] = {"arch": best_arch, "wape": best_wape}
        return best
    sum_err, sum_act = 0.0, 0.0
    for cid in CLIENT_ORDER:
        if cid not in m:
            continue
        nodp = m[cid]["nodp"]
        n = nodp["n"]
        sum_act += nodp["mae"] * n / nodp["wape"] if nodp["wape"] > 0 else 0
        best_wape, best_arch = float("inf"), None
        for a in ARCHS:
            rc = metrics[e][a]["per_client"][cid].get("dp+rc")
            if rc and rc["wape"] < best_wape:
                best_wape, best_arch = rc["wape"], a
        if best_arch:
            sum_err += (metrics[e][best_arch]["per_client"][cid]
                        ["dp+rc"]["mae"] * n)
    return (sum_err / sum_act * 100) if sum_act > 0 else float("nan")


def _agg_series(metrics):
    """Aggregate WAPE (%) series: nodp constant, dp, dp+rc(best) per ε."""
    nodp = metrics[EPS[0]]["mlp"]["aggregate"]["nodp"]["wape"] * 100
    dp = [metrics[e]["mlp"]["aggregate"]["dp"]["wape"] * 100 for e in EPS]
    best = [compute_best_rc(metrics, e) for e in EPS]
    return nodp, dp, best


# ============================================================================
# Fig 1  privacy–utility trade-off (core figure)
# ============================================================================
def fig1(metrics):
    nodp, dp, best = _agg_series(metrics)
    x = [float(e) for e in EPS]
    span = max(dp) - min(best)

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.axhline(nodp, color=C["nodp"], ls="--", lw=1.4, alpha=0.85, zorder=1)
    ax.fill_between(x, [nodp] * len(x), dp, color=C["dp"], alpha=0.06, zorder=0)
    ax.fill_between(x, dp, best, color=C["dp+rc"], alpha=0.10, zorder=0)

    ax.plot(x, dp, "o-", color=C["dp"], lw=1.6, ms=4.0, zorder=3)
    ax.plot(x, best, "s-", color=C["dp+rc"], lw=1.6, ms=4.0, zorder=3)

    d_off = 0.12 * span
    for xi, v in zip(x, dp):
        ax.text(xi, v + 0.08 * span, f"{v:.2f}", ha="center", va="bottom",
                fontsize=6.3, color=C["dp"])
    for xi, v in zip(x, best):
        ax.text(xi, v - d_off, f"{v:.2f}", ha="center", va="top",
                fontsize=6.3, color=C["dp+rc"])

    # Explicit end labels (no legend needed)
    ax.text(7.9, nodp + 0.30 * span, "nodp (no DP)", color=C["nodp"],
            fontsize=7, va="bottom", ha="left")
    ax.text(7.9, dp[-1], "dp", color=C["dp"], fontsize=7, va="center", ha="left")
    ax.text(7.9, best[-1], "dp+rc (best)", color=C["dp+rc"], fontsize=7,
            va="center", ha="left")

    # Representative gaps at the strongest privacy setting (ε = 0.5)
    ax.annotate("", xy=(0.78, dp[0]), xytext=(0.78, nodp),
                arrowprops=dict(arrowstyle="<->", color=C["dp"], lw=1.0,
                                shrinkA=0, shrinkB=0))
    ax.text(0.95, nodp + 0.20 * span, f"DP cost\n+{dp[0]-nodp:.2f} pp",
            color=C["dp"], fontsize=6.4, va="bottom", ha="left")
    ax.annotate("", xy=(1.24, dp[0] - 0.10 * span),
                xytext=(1.24, best[0] - 0.10 * span),
                arrowprops=dict(arrowstyle="<->", color=C["dp+rc"], lw=1.0,
                                shrinkA=0, shrinkB=0))
    ax.text(1.42, (dp[0] + best[0]) / 2 - 0.10 * span,
            f"RC gain\n-{dp[0]-best[0]:.2f} pp", color=C["dp+rc"],
            fontsize=6.4, va="center", ha="left")

    ax.set_xlabel("Privacy budget ε")
    ax.set_ylabel("Aggregate WAPE (%)")
    ax.set_xlim(0.2, 8.75)
    lo = min(nodp, min(best)) - 0.35 * span
    hi = max(dp) + 0.55 * span
    ax.set_ylim(lo, hi)
    ax.set_xticks(x)
    ax.grid(axis="y")
    footnote(fig, FOOT + "; WAPE = Σ|ŷ−y|/Σ|y|")
    save(fig, FIGS / "fig_privacy_utility.png")


# ============================================================================
# Fig 2  three-mode performance + RC gain per ε
# ============================================================================
def fig2(metrics):
    nodp, dp, best = _agg_series(metrics)
    x = np.arange(len(EPS))
    w = 0.30                     # bars nearly touching (thin white seams)
    line_h = 0.09 * max(dp)

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.bar(x - w, [nodp] * len(EPS), w, color=C["nodp"],
           label="nodp (no DP)", edgecolor="white", lw=0.5)
    ax.bar(x, dp, w, color=C["dp"], label="dp",
           edgecolor="white", lw=0.5)
    ax.bar(x + w, best, w, color=C["dp+rc"], label="dp+rc (best)",
           edgecolor="white", lw=0.5)

    for i in range(len(EPS)):
        ax.text(x[i], dp[i] + 0.02 * max(dp), f"{dp[i]:.2f}", ha="center",
                va="bottom", fontsize=6.2, style="italic", color="#333333")
        ax.text(x[i], dp[i] + line_h, f"+{dp[i]-nodp:.2f}pp",
                ha="center", va="bottom", fontsize=5.8, style="italic",
                color="#777777")
        ax.text(x[i] + w, best[i] + 0.02 * max(dp), f"{best[i]:.2f}",
                ha="center", va="bottom", fontsize=6.2, style="italic",
                color="#333333")
        ax.text(x[i] + w, best[i] + line_h, f"−{dp[i]-best[i]:.2f}pp",
                ha="center", va="bottom", fontsize=5.8, style="italic",
                color="#777777")

    ax.set_xticks(x)
    ax.set_xticklabels([f"ε={e}" for e in EPS])
    ax.set_ylabel("Aggregate WAPE (%)")
    ax.set_xlim(-0.62, len(EPS) - 0.38)
    ax.set_ylim(0, max(dp) + 2.4 * line_h)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3,
              columnspacing=1.6)
    ax.grid(axis="y", zorder=2.5, color="#b0b0b0", lw=0.6, alpha=0.9)
    footnote(fig, FOOT)
    save(fig, FIGS / "fig_rc_ablation.png")


# ============================================================================
# Fig 3  per-client detail (one figure per ε)
# ============================================================================
def fig3(metrics):
    for e in EPS:
        data = metrics[e]
        m = data["mlp"]["per_client"]
        best = compute_best_rc(metrics, e, per_client=True)
        cids = [c for c in CLIENT_ORDER if c in m]
        x = np.arange(len(cids))
        w = 0.30                     # bars nearly touching (thin white seams)

        nodp_vals = [m[c]["nodp"]["wape"] * 100 for c in cids]
        dp_vals = [m[c]["dp"]["wape"] * 100 for c in cids]
        rc_vals = [best[c]["wape"] * 100 if c in best else float("nan")
                   for c in cids]

        fig, ax = plt.subplots(figsize=(8.8, 3.6))
        ax.bar(x - w, nodp_vals, w, color=C["nodp"], label="nodp",
               edgecolor="white", lw=0.5)
        ax.bar(x, dp_vals, w, color=C["dp"], label="dp",
               edgecolor="white", lw=0.5)
        ax.bar(x + w, rc_vals, w, color=C["dp+rc"], label="dp+rc (best)",
               edgecolor="white", lw=0.5)

        vmax = max(dp_vals + nodp_vals + rc_vals)
        line_h = 0.045 * vmax
        for i in range(len(cids)):
            ax.text(x[i], dp_vals[i] + 0.4 * line_h, f"{dp_vals[i]:.1f}",
                    ha="center", va="bottom", fontsize=5.8, style="italic",
                    color="#333333")
            ax.text(x[i] + w, rc_vals[i] + 0.4 * line_h, f"{rc_vals[i]:.1f}",
                    ha="center", va="bottom", fontsize=5.8, style="italic",
                    color="#333333")
            if best[cids[i]]["arch"]:
                # arch label sits INSIDE the green bar, just below its top edge
                ax.text(x[i] + w, rc_vals[i] - 0.35 * line_h,
                        best[cids[i]]["arch"], ha="center", va="top",
                        fontsize=5.4, color="#ffffff", style="italic")

        # Dataset group separators + light-gray type labels (top row)
        offset = 0
        for ds, n in DATASETS:
            offset += n
            if offset < len(cids):
                ax.axvline(offset - 0.5, color="#bbbbbb", lw=0.6, ls=":")
            ax.text(offset - n / 2 - 0.5, vmax + 3.3 * line_h, DS_LABEL[ds],
                    ha="center", va="bottom", fontsize=6.3, color="#999999")

        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", " ") for c in cids],
                           rotation=35, ha="right", fontsize=6.4)
        ax.set_ylabel("WAPE (%)")
        ax.set_xlim(-0.62, len(cids) - 0.38)
        ax.set_ylim(0, vmax + 4.6 * line_h)
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.87), fontsize=6.4)
        ax.grid(axis="y", zorder=2.5, color="#b0b0b0", lw=0.6, alpha=0.9)
        fig.suptitle(f"Per-client WAPE (ε = {e}, de-normalised)",
                     fontsize=8.5, x=0.005, ha="left")
        footnote(fig, "italic label inside green bar = client's best RC architecture")
        save(fig, EPS_FIG_ROOT / f"epsilon-{e}"
             / f"fig_per_client_epsilon_{e}.png")


# ============================================================================
# Fig 4  training convergence
# ============================================================================
def fig4():
    nodp = _load_json(BASELINE / "no-dp" / "baseline_history.json")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(nodp["train_losses"], color=C["nodp"], ls="--", lw=1.5,
            label="nodp", zorder=4)
    for e in EPS:
        d = _load_json(BASELINE / "dp" / f"epsilon-{e}" / "baseline_history.json")
        ax.plot(d["train_losses"], color=EPS_COLORS[e], lw=1.3,
                marker="o", markevery=5, ms=2.8, label=f"dp ε={e}",
                zorder=3)

    top = max(nodp["train_losses"])
    for e in EPS:
        d = _load_json(BASELINE / "dp" / f"epsilon-{e}" / "baseline_history.json")
        top = max(top, max(d["train_losses"]))
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Training loss (normalised MAE)")
    ax.set_ylim(0, top * 1.03)
    ax.set_xlim(0, 30.5)
    ax.legend(loc="upper left", fontsize=6.6)
    ax.grid(axis="y")
    footnote(fig, "Normalised-space loss; per-round aggregate across 9 clients")
    save(fig, FIGS / "fig_convergence.png")


# ============================================================================
# Fig 5  privacy audit (3 panels)
# ============================================================================
def fig5():
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.0))

    # (A) participation: all 5 ε overlap at 9/9 for all 30 rounds
    ax = axes[0]
    audits = [(_load_json(BASELINE / "dp" / f"epsilon-{e}" / "audit_log.json"))
              for e in EPS]
    rounds = [r["round"] for r in audits[0]["rounds"]]
    expected = len(audits[0]["rounds"][0]["expected"])
    for a, col in zip(audits, [EPS_COLORS[e] for e in EPS]):
        joined = [len(r["joined"]) for r in a["rounds"]]
        ax.plot(rounds, joined, "-", color=col, lw=1.1, alpha=0.75)
    ax.axhline(expected, color="#999999", ls="--", lw=0.9)
    ax.annotate(f"30 rounds: {expected}/{expected} joined,\n0 dropped (all ε)",
                xy=(30, expected), xytext=(16, expected - 2.2),
                fontsize=6.2, color="#333333",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#888888"))
    ax.set_title("(A) Participation per round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Clients")
    ax.set_ylim(0, expected + 1.5)
    ax.set_yticks([0, 3, 6, 9])
    ax.grid(axis="y")

    # (B) per-client ε spent (ε = 0.5)
    ax = axes[1]
    cfg = _load_json(BASELINE / "dp" / "epsilon-0.5" / "config.json")
    pc = cfg["dp"]["per_client"]
    cids = [cid for cid in CLIENT_ORDER if cid in pc]
    spent = [pc[cid]["epsilon"] for cid in cids]
    bars = ax.bar(range(len(cids)), spent, width=0.62, color=C["dp"],
                  alpha=0.85)
    lo = min(spent) - 0.0016
    hi = max(spent) + 0.0016
    for i, v in enumerate(spent):
        ax.text(i, v + 0.0001, f"{v:.4f}", ha="center", va="bottom",
                fontsize=5.6)
    ax.axhline(0.5, color="#999999", ls="--", lw=0.9)
    ax.set_title("(B) Per-client ε spent (strongest privacy)")
    ax.set_xticks(range(len(cids)))
    ax.set_xticklabels([c.replace("_", " ") for c in cids],
                       rotation=30, ha="right", fontsize=5.8)
    ax.set_ylim(lo, hi)
    ax.set_yticks([0.5, 0.5005, 0.501])
    ax.grid(axis="y", alpha=0.4)
    ax.text(0.02, 0.98, "deviation < 0.001 (PLD accounting)",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.8,
            color="#555555")

    # (C) adaptive clipping norm — distinct color per ε budget
    ax = axes[2]
    for e in EPS:
        a = _load_json(BASELINE / "dp" / f"epsilon-{e}" / "audit_log.json")
        clip = [r.get("clip_norm") for r in a["rounds"]]
        if clip and clip[0] is not None:
            ax.plot(rounds, clip, "-", color=EPS_DISTINCT[e], lw=1.3,
                    label=f"ε={e}")
    ax.set_title("(C) Adaptive clip norm C")
    ax.set_xlabel("Round")
    ax.set_ylabel("Clip norm C")
    ax.legend(fontsize=6, loc="lower right")
    ax.grid(axis="y")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    footnote(fig, "Source: server audit_log (round-by-round) and config.json")
    save(fig, FIGS / "fig_privacy_audit.png")


# ============================================================================
# Table 6  summary table (Markdown)
# ============================================================================
def table6(metrics):
    nodp, dp, best = _agg_series(metrics)
    head = "| Variant | " + " | ".join(f"ε={e}" for e in EPS) + " |"
    sep = "|" + "---|" * (len(EPS) + 1)

    def row(name, vals):
        return f"| {name} | " + " | ".join(f"{v:.2f}%" for v in vals) + " |"

    lines = [head, sep,
             row("No-DP", [nodp] * len(EPS)),
             row("DP", dp),
             row("DP+RC (best)", best),
             f"| DP cost (dp − nodp) | "
             + " | ".join(f"{d-nodp:+.2f} pp" for d in dp) + " |",
             f"| RC gain (dp+rc − dp) | "
             + " | ".join(f"{r-d:+.2f} pp" for r, d in zip(best, dp)) + " |"]
    md = "\n".join(lines)
    with open(FIGS / "summary_table.md", "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print("   saved summary_table.md")
    print(md)


# ============================================================================
# Fig 7  RC architecture ablation
# ============================================================================
def fig7(metrics):
    _, _, best = _agg_series(metrics)
    arch_vals = {a: [metrics[e][a]["aggregate"]["dp+rc"]["wape"] * 100
                     for e in EPS] for a in ARCHS}
    x = np.arange(len(EPS))
    w = 0.215                    # bars nearly touching (thin white seams)

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for i, a in enumerate(ARCHS):
        ax.bar(x + (i - 1.5) * w, arch_vals[a], w, color=ARCH_C[a],
               label=a, edgecolor="white", lw=0.5)
    ax.bar(x + 1.5 * w, best, w, color=ARCH_C["best"],
           label="best (per-client)", edgecolor="white", lw=0.5)

    for i in range(len(EPS)):
        ax.text(x[i] + 1.5 * w, best[i] + 0.03 * max(best), f"{best[i]:.2f}",
                ha="center", va="bottom", fontsize=6.2, style="italic",
                color="#333333")

    gains = [min(arch_vals[a][i] for a in ARCHS) - best[i]
             for i in range(len(EPS))]
    ax.set_xticks(x)
    ax.set_xticklabels([f"ε={e}" for e in EPS])
    ax.set_ylabel("Aggregate WAPE (%)")
    ax.set_xlim(-0.55, len(EPS) - 0.45)
    ax.set_ylim(0, max(max(arch_vals[a]) for a in ARCHS) * 1.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=4,
              columnspacing=1.4, fontsize=6.4)
    ax.grid(axis="y", zorder=2.5, color="#b0b0b0", lw=0.6, alpha=0.9)
    footnote(fig, FOOT + f"; best beats best fixed arch by "
             f"{sum(gains)/len(gains):.2f} pp on average")
    save(fig, FIGS / "fig_rc_arch_ablation.png")


# ============================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig-root", type=Path, default=None,
                        help="preview root: figures go to <fig-root>/ instead "
                             "of the official analysis/ dirs")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    setup_style()
    if args.fig_root is not None:
        global FIGS, EPS_FIG_ROOT
        FIGS = args.fig_root / "figs"
        EPS_FIG_ROOT = args.fig_root
        print(f"preview mode: outputs written to {args.fig_root}")
    metrics = load_all_metrics()
    print(f"loaded {sum(len(v) for v in metrics.values())} JSON files\n")
    print("Fig 1 privacy-utility"); fig1(metrics)
    print("Fig 2 mode ablation"); fig2(metrics)
    print("Fig 3 per-client"); fig3(metrics)
    print("Fig 4 convergence"); fig4()
    print("Fig 5 privacy audit"); fig5()
    print("Table 6 summary"); table6(metrics)
    print("Fig 7 RC arch ablation"); fig7(metrics)
    print("\n=== all figures generated ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())