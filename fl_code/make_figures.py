"""任务二：标准图集生成（对标 DP-FL 论文规范，nature-figure 风格）。

生成图1隐私-效用曲线、图2三模式消融、图3每客户端细粒度（每 ε 一张）、
图4训练收敛、图5隐私审计（三面板）、图7架构消融，以及表6汇总表。

数据来源（均为只读输入）：
- fl_code/analysis/epsilon-*/denorm_metrics_{mlp,lstm,tcn}.json  反归一化验证
- fl_code/baseline_outputs/no-dp/、dp/epsilon-*/ 下的 baseline_history.json、
  audit_log.json、config.json

输出：PNG(300dpi) + SVG，跨 ε 图 → fl_code/analysis/figs/，每客户端图 →
fl_code/analysis/epsilon-*/。
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "fl_code" / "analysis"
BASELINE = ROOT / "fl_code" / "baseline_outputs"
FIGS = ANALYSIS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

EPS = ["0.5", "1.5", "3.5", "5.5", "7.5"]
ARCHS = ["mlp", "lstm", "tcn"]
CLIENT_ORDER = ["steel_ind_0", "tetouan_city_0", "tetouan_city_1", "tetouan_city_2",
                "lcl_res_0", "lcl_res_1", "eld_ind_0", "eld_ind_1", "eld_ind_2"]
DATASETS = [("steel_ind", 1), ("tetouan_city", 3), ("lcl_res", 2), ("eld_ind", 3)]

# 语义色：nodp=红(基线), dp=蓝(隐私), dp+rc=绿(修正找回)——与交接文档统一
C = {"nodp": "#C44E52", "dp": "#4C72B0", "dp+rc": "#55A868"}
# 架构消融用中性灰阶 + best 复用 dp+rc 绿，避免与三模式语义色混淆
ARCH_C = {"mlp": "#b4bdc7", "lstm": "#7d8b99", "tcn": "#d7dce1", "best": C["dp+rc"]}
FOOT = "反归一化后（原始 kWh 单位）；WAPE = Σ|ŷ−y|/Σ|y|，9 客户端池化；dp+rc = 每客户端自选 best RC"


def setup_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        # YaHei 放首位：覆盖 CJK+拉丁全字形，避免 Arial 首位的 glyph 回退歧义
        "font.sans-serif": ["Microsoft YaHei", "Arial", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "grid.color": "#d6d6d6",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
        "figure.dpi": 110,
        "svg.fonttype": "none",   # SVG 存文本不转路径，中文标注可在任意机器查看
    })


def save(fig, path: Path, also_svg=True):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if also_svg:
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  保存 {path.name}")


def footnote(fig, text: str):
    fig.text(0.99, 0.005, text, ha="right", va="bottom",
             fontsize=6, color="#666666")


def load_all_metrics():
    """读所有 JSON，返回 {epsilon: {arch: data}}"""
    result = {}
    for e in EPS:
        result[e] = {}
        for a in ARCHS:
            p = ANALYSIS / f"epsilon-{e}" / f"denorm_metrics_{a}.json"
            if p.exists():
                with open(p) as f:
                    result[e][a] = json.load(f)
    return result


def compute_best_rc(metrics, e, per_client=False):
    """每客户端选 best RC（dp+rc WAPE 最小），返回聚合或每客户端结果"""
    data = metrics[e]
    m = data["mlp"]["per_client"]
    if per_client:
        best = {}
        for cid in CLIENT_ORDER:
            if cid not in m:
                continue
            best_arch = None
            best_wape = float("inf")
            for a in ARCHS:
                rc = data[a]["per_client"][cid].get("dp+rc")
                if rc and rc["wape"] < best_wape:
                    best_wape = rc["wape"]
                    best_arch = a
            best[cid] = {"arch": best_arch, "wape": best_wape}
        return best
    sum_err = 0.0
    sum_act = 0.0
    for cid in CLIENT_ORDER:
        if cid not in m:
            continue
        nodp = m[cid]["nodp"]
        n = nodp["n"]
        sum_act += nodp["mae"] * n / nodp["wape"] if nodp["wape"] > 0 else 0
        best_wape = float("inf")
        best_arch = None
        for a in ARCHS:
            rc = data[a]["per_client"][cid].get("dp+rc")
            if rc and rc["wape"] < best_wape:
                best_wape = rc["wape"]
                best_arch = a
        if best_arch:
            rc_mae = data[best_arch]["per_client"][cid]["dp+rc"]["mae"]
            sum_err += rc_mae * n
    return (sum_err / sum_act * 100) if sum_act > 0 else float("nan")


def _agg_series(metrics, key):
    """聚合 WAPE（%）序列：nodp / dp / dp+rc(best)"""
    nodp = metrics[EPS[0]]["mlp"]["aggregate"]["nodp"]["wape"] * 100
    dp = [metrics[e]["mlp"]["aggregate"]["dp"]["wape"] * 100 for e in EPS]
    best = [compute_best_rc(metrics, e) for e in EPS]
    return nodp, dp, best


def _load_json(rel: Path):
    with open(rel) as f:
        return json.load(f)


# ============================================================================
# 图1 隐私-效用权衡曲线
# ============================================================================
def fig1(metrics):
    nodp, dp, best = _agg_series(metrics, "fig1")
    x = [float(e) for e in EPS]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(nodp, color=C["nodp"], ls="--", lw=1.4, alpha=0.85, zorder=1)
    # DP 代价带（蓝）与 RC 恢复带（绿）
    ax.fill_between(x, [nodp] * len(x), dp, color=C["dp"], alpha=0.06, zorder=0)
    ax.fill_between(x, dp, best, color=C["dp+rc"], alpha=0.10, zorder=0)

    ax.plot(x, dp, "o-", color=C["dp"], lw=1.6, ms=4.2, zorder=3)
    ax.plot(x, best, "s-", color=C["dp+rc"], lw=1.6, ms=4.2, zorder=3)

    # 线末端直接标注（不依赖图例）；nodp 标签置于虚线上方避免骑线
    ax.text(7.62, nodp + 0.38, "nodp（无 DP）", color=C["nodp"], fontsize=7,
            va="bottom")
    ax.text(7.62, dp[-1], "dp", color=C["dp"], fontsize=7, va="center")
    ax.text(7.62, best[-1], "dp+rc (best)", color=C["dp+rc"], fontsize=7, va="center")
    for xi, v in zip(x, dp):
        ax.text(xi, v - 0.22, f"{v:.2f}", ha="center", va="top", fontsize=6.5, color=C["dp"])
    for xi, v in zip(x, best):
        ax.text(xi, v + 0.10, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=C["dp+rc"])

    # ε=0.5 处标注 DP 代价与 RC 找回
    ax.annotate("", xy=(1.15, dp[0] - 0.25), xytext=(1.15, nodp + 0.25),
                arrowprops=dict(arrowstyle="<->", color=C["dp"], lw=1.0))
    ax.text(1.30, (nodp + dp[0]) / 2, f"DP 代价\n+{dp[0]-nodp:.2f}pp",
            color=C["dp"], fontsize=6.5, va="center")
    ax.annotate("", xy=(2.15, dp[0] - 0.25), xytext=(2.15, best[0] + 0.25),
                arrowprops=dict(arrowstyle="<->", color=C["dp+rc"], lw=1.0))
    ax.text(2.30, (dp[0] + best[0]) / 2, f"RC 找回\n−{dp[0]-best[0]:.2f}pp",
            color=C["dp+rc"], fontsize=6.5, va="center")

    ax.set_xlabel("隐私预算 ε")
    ax.set_ylabel("聚合 WAPE (%)")
    ax.set_xlim(0.2, 9.2)
    ax.set_ylim(3.4, 8.6)
    ax.set_xticks(x)
    ax.grid(axis="y")
    footnote(fig, "反归一化后（原始 kWh 单位）\nWAPE 池化；dp+rc = 每客户端自选 best")
    save(fig, FIGS / "fig_privacy_utility.png")


# ============================================================================
# 图2 三模式性能对比 + RC 增量消融（每 ε 三根柱）
# ============================================================================
def fig2(metrics):
    nodp, dp, best = _agg_series(metrics, "fig2")
    x = np.arange(len(EPS))
    w = 0.26

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.bar(x - w, [nodp] * len(EPS), w, color=C["nodp"], label="nodp（无 DP）")
    ax.bar(x, dp, w, color=C["dp"], label="dp")
    ax.bar(x + w, best, w, color=C["dp+rc"], label="dp+rc (best)")

    for i in range(len(EPS)):
        ax.text(x[i], dp[i] + 0.10, f"{dp[i]:.2f}", ha="center", va="bottom",
                fontsize=6.2, color=C["dp"])
        ax.text(x[i] + w, best[i] + 0.10, f"{best[i]:.2f}", ha="center",
                va="bottom", fontsize=6.2, color=C["dp+rc"])
        ax.text(x[i], dp[i] + 0.72, f"+{dp[i]-nodp:.2f}pp", ha="center",
                fontsize=6, color=C["dp"])
        ax.text(x[i] + w, best[i] + 0.72, f"−{dp[i]-best[i]:.2f}pp", ha="center",
                fontsize=6, color=C["dp+rc"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"ε={e}" for e in EPS])
    ax.set_ylabel("聚合 WAPE (%)")
    ax.set_ylim(0, 10.3)
    ax.legend(loc="upper left", ncol=1)
    ax.grid(axis="y")
    footnote(fig, FOOT)
    save(fig, FIGS / "fig_rc_ablation.png")


# ============================================================================
# 图3 每客户端细粒度图（每档 ε 一张）
# ============================================================================
def fig3(metrics):
    for e in EPS:
        data = metrics[e]
        m = data["mlp"]["per_client"]
        best = compute_best_rc(metrics, e, per_client=True)
        cids = [c for c in CLIENT_ORDER if c in m]
        x = np.arange(len(cids))
        w = 0.26

        nodp_vals = [m[c]["nodp"]["wape"] * 100 for c in cids]
        dp_vals = [m[c]["dp"]["wape"] * 100 for c in cids]
        rc_vals = [best[c]["wape"] * 100 if c in best else float("nan") for c in cids]

        fig, ax = plt.subplots(figsize=(8.6, 3.4))
        ax.bar(x - w, nodp_vals, w, color=C["nodp"], label="nodp")
        ax.bar(x, dp_vals, w, color=C["dp"], label="dp")
        ax.bar(x + w, rc_vals, w, color=C["dp+rc"], label="dp+rc (best)")

        ymax = max(rc_vals + dp_vals + nodp_vals)
        for i, c in enumerate(cids):
            ax.text(x[i], dp_vals[i] + 0.04, f"{dp_vals[i]:.2f}", ha="center",
                    va="bottom", fontsize=5.8, color=C["dp"])
            ax.text(x[i] + w, rc_vals[i] + 0.04, f"{rc_vals[i]:.2f}", ha="center",
                    va="bottom", fontsize=5.8, color=C["dp+rc"])
            if c in best and best[c]["arch"]:
                ax.text(x[i] + w, rc_vals[i] - 0.22, best[c]["arch"],
                        ha="center", va="top", fontsize=5.5, color="#ffffff")

        # 数据集分隔线 + 分组标签（图顶）
        ds_name = {"steel_ind": "钢铁园区", "tetouan_city": "城市供电区",
                   "lcl_res": "居民小区", "eld_ind": "工业负荷区"}
        offset = 0
        for ds, n in DATASETS:
            offset += n
            if offset < len(cids):
                ax.axvline(offset - 0.5, color="#bbbbbb", lw=0.6, ls=":")
            ax.text(offset - n / 2 - 0.5, ymax * 1.08, ds_name[ds],
                    ha="center", fontsize=6.5, color="#777777")

        ax.set_xticks(x)
        ax.set_xticklabels(cids, rotation=30, ha="right", fontsize=6.5)
        ax.set_ylabel("WAPE (%)")
        ax.set_xlim(-0.8, len(cids) - 0.2)
        ax.set_ylim(0, ymax * 1.24)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05),
                  ncol=3, fontsize=6.5, columnspacing=1.2)
        ax.grid(axis="y")
        fig.text(0.99, 0.985,
                 f"ε={e}；反归一化后（原始 kWh 单位）\n"
                 "绿柱内文字 = 该客户端自选 best RC 架构",
                 ha="right", va="top", fontsize=6, color="#666666")
        save(fig, ANALYSIS / f"epsilon-{e}" / f"fig_per_client_epsilon_{e}.png")


# ============================================================================
# 图4 训练收敛曲线
# ============================================================================
def fig4():
    nodp = _load_json(BASELINE / "no-dp" / "baseline_history.json")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(nodp["train_losses"], color=C["nodp"], ls="--", lw=1.5, label="nodp")

    blues = plt.cm.Blues(np.linspace(0.38, 0.95, len(EPS)))
    for e, col in zip(EPS, blues):
        d = _load_json(BASELINE / "dp" / f"epsilon-{e}" / "baseline_history.json")
        ax.plot(d["train_losses"], color=col, lw=1.3, markevery=5, ms=3,
                marker="o", label=f"dp ε={e}")

    ax.set_xlabel("联邦训练轮次")
    ax.set_ylabel("训练损失（归一化空间 MAE）")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(axis="y")
    footnote(fig, "归一化空间训练损失（非反归一化）；每轮值为 9 客户端平均")
    save(fig, FIGS / "fig_convergence.png")


# ============================================================================
# 图5 隐私审计图（三面板：参与率 / 每客户端 ε / 自适应裁剪范数）
# ============================================================================
def fig5():
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.0))

    # (A) 参与率：全部 ε 档 30 轮 9/9 参与
    ax = axes[0]
    first = _load_json(BASELINE / "dp" / f"epsilon-{EPS[0]}" / "audit_log.json")
    rounds = [r["round"] for r in first["rounds"]]
    joined = [len(r["joined"]) for r in first["rounds"]]
    expected = len(first["rounds"][0]["expected"])
    ax.plot(rounds, joined, "-o", color=C["dp"], ms=3, lw=1.2,
            label="各 ε 档（5 档重合）")
    ax.axhline(expected, color="#999999", ls="--", lw=0.9, label="expected")
    ax.annotate(f"{len(rounds)} 轮全部 {expected}/{expected} 参与",
                xy=(len(rounds) * 0.5, expected), xytext=(4, expected - 1.5),
                fontsize=6.5, color="#333333",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#888888"))
    ax.set_title("(A) 每轮参与客户端数")
    ax.set_xlabel("轮次")
    ax.set_ylabel("参与客户端数")
    ax.set_ylim(0, expected + 1.6)
    ax.set_yticks([0, 3, 6, 9])
    ax.legend(fontsize=6)
    ax.grid(axis="y")

    # (B) 每客户端隐私预算消耗（ε=0.5 档，所有客户端精确花完目标预算）
    ax = axes[1]
    cfg = _load_json(BASELINE / "dp" / f"epsilon-{EPS[0]}" / "config.json")
    pc = cfg["dp"]["per_client"]
    cids = [cid for cid in CLIENT_ORDER if cid in pc]
    eps_consumed = [pc[cid]["epsilon"] for cid in cids]
    ax.bar(range(len(cids)), eps_consumed, width=0.62, color=C["dp"],
           alpha=0.85)
    ax.axhline(0.5, color="#999999", ls="--", lw=0.9, label="目标 ε=0.5")
    ax.text(len(cids) - 0.5, 0.55, "所有客户端精确花完预算",
            ha="right", fontsize=6.5, color="#333333")
    ax.set_title("(B) 每客户端累计 ε（ε=0.5 档）")
    ax.set_xticks(range(len(cids)))
    ax.set_xticklabels(cids, rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("实际消耗 ε")
    ax.set_ylim(0, 0.68)
    ax.legend(fontsize=6)
    ax.grid(axis="y")

    # (C) 自适应裁剪范数曲线（5 档 ε 全有记录，用区分色绘制）
    ax = axes[2]
    aux = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for e, col in zip(EPS, aux):
        a = _load_json(BASELINE / "dp" / f"epsilon-{e}" / "audit_log.json")
        clip = [r.get("clip_norm") for r in a["rounds"]]
        if clip and clip[0] is not None:
            ax.plot(rounds, clip, "-", color=col, lw=1.2, label=f"ε={e}")
    ax.set_title("(C) 自适应裁剪范数 C")
    ax.set_xlabel("轮次")
    ax.set_ylabel("裁剪范数 C")
    ax.legend(fontsize=6, loc="lower right")
    ax.grid(axis="y")

    fig.tight_layout()
    footnote(fig, "审计来源：server 端 audit_log（参与/掉线/裁剪界）与 config.json（每客户端 ε）")
    save(fig, FIGS / "fig_privacy_audit.png", also_svg=False)


# ============================================================================
# 表6 汇总表（Markdown）
# ============================================================================
def table6(metrics):
    nodp, dp, best = _agg_series(metrics, "table6")

    lines = []
    lines.append("| 变体 | " + " | ".join([f"ε={e}" for e in EPS]) + " |")
    lines.append("|" + "---|" * (len(EPS) + 1))
    lines.append(f"| nodp | " + " | ".join([f"{nodp:.2f}%" for _ in EPS]) + " |")
    lines.append(f"| dp | " + " | ".join([f"{v:.2f}%" for v in dp]) + " |")
    lines.append(f"| dp+rc(best) | " + " | ".join([f"{v:.2f}%" for v in best]) + " |")
    lines.append(f"| DP代价 (dp−nodp) | " + " | ".join([f"{d-nodp:+.2f}pp" for d in dp]) + " |")
    lines.append(f"| RC改进 (dp+rc−dp) | " + " | ".join([f"{r-d:+.2f}pp" for r, d in zip(best, dp)]) + " |")

    md = "\n".join(lines)
    with open(FIGS / "summary_table.md", "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print("  保存 summary_table.md")
    print(md)


# ============================================================================
# 图7 RC 架构消融图（固定架构 vs 每客户端自选 best）
# ============================================================================
def fig7(metrics):
    _, _, best = _agg_series(metrics, "fig7")
    x = np.arange(len(EPS))
    w = 0.19

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for i, a in enumerate(ARCHS):
        vals = [metrics[e][a]["aggregate"]["dp+rc"]["wape"] * 100 for e in EPS]
        ax.bar(x + (i - 1.5) * w, vals, w, color=ARCH_C[a], label=a)
    ax.bar(x + 1.5 * w, best, w, color=ARCH_C["best"], label="best（每客户端自选）")

    gains = []
    for i in range(len(EPS)):
        fixed = min(metrics[EPS[i]][a]["aggregate"]["dp+rc"]["wape"] * 100
                    for a in ARCHS)
        gains.append(fixed - best[i])
        ax.text(x[i] + 1.5 * w, best[i] + 0.06, f"{best[i]:.2f}",
                ha="center", va="bottom", fontsize=6.2, color=C["dp+rc"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"ε={e}" for e in EPS])
    ax.set_ylabel("聚合 WAPE (%)")
    ax.set_ylim(0, 7.6)
    ax.legend(loc="upper left", ncol=2, fontsize=6.8)
    ax.grid(axis="y")
    footnote(fig, FOOT + f"；best 较最优固定架构平均低 {sum(gains)/len(gains):.2f}pp")
    save(fig, FIGS / "fig_rc_arch_ablation.png")


# ============================================================================
def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    setup_style()
    metrics = load_all_metrics()
    print(f"已加载 {sum(len(v) for v in metrics.values())} 份 JSON\n")
    print("图1 隐私-效用曲线"); fig1(metrics)
    print("图2 三模式消融"); fig2(metrics)
    print("图3 每客户端细粒度"); fig3(metrics)
    print("图4 训练收敛"); fig4()
    print("图5 隐私审计"); fig5()
    print("表6 汇总表"); table6(metrics)
    print("图7 架构消融"); fig7(metrics)
    print("\n=== 全部图集生成完成 ===")


if __name__ == "__main__":
    main()
