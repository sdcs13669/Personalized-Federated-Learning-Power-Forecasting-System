"""任务二：标准图集生成（对标 DP-FL 论文规范）。
生成图1隐私-效用曲线、图2消融、图3每客户端、图4收敛、图5审计、图7架构消融、表6汇总表。
"""
import json
import os
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

# 配色（nodp=红, dp=蓝, dp+rc=绿）
C = {"nodp": "#d62728", "dp": "#1f77b4", "dp+rc": "#2ca02c"}

# 设置中文字体
def setup_font():
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"):
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False

CN = setup_font()

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
    """每客户端选 best RC（dp+rc wape 最小），返回聚合或每客户端结果"""
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
    else:
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

# ============================================================================
# 图1 隐私-效用权衡曲线
# ============================================================================
def fig1(metrics):
    nodp = metrics[EPS[0]]["mlp"]["aggregate"]["nodp"]["wape"] * 100
    dp = [metrics[e]["mlp"]["aggregate"]["dp"]["wape"] * 100 for e in EPS]
    best_rc = [compute_best_rc(metrics, e) for e in EPS]
    x = [float(e) for e in EPS]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(nodp, color=C["nodp"], ls="--", lw=2, label=f"nodp (无DP) = {nodp:.2f}%")
    ax.plot(x, dp, "o-", color=C["dp"], lw=2, label="dp (有DP)")
    ax.plot(x, best_rc, "s-", color=C["dp+rc"], lw=2, label="dp+rc (best)")

    # 标注 ε=0.5 的 RC 找回
    gap = dp[0] - best_rc[0]
    ax.annotate(f"ε=0.5: RC找回 {gap:.2f}pp",
                xy=(x[0], best_rc[0]), xytext=(x[0]+0.4, best_rc[0]+0.6),
                arrowprops=dict(arrowstyle="->"), fontsize=9)

    ax.set_xlabel("隐私预算 ε")
    ax.set_ylabel("聚合 WAPE (%)")
    ax.set_title("隐私-效用权衡曲线（反归一化后，原始 kWh 单位）" if CN else "Privacy-Utility Tradeoff")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.text(0.99, 0.01, "注：反归一化后（原始 kWh 单位）", ha="right", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(FIGS / "fig_privacy_utility.png", dpi=150)
    plt.close()
    print("图1 保存:", FIGS / "fig_privacy_utility.png")

# ============================================================================
# 图2 三模式性能对比 + RC 增量消融
# ============================================================================
def fig2(metrics):
    nodp = metrics[EPS[0]]["mlp"]["aggregate"]["nodp"]["wape"] * 100
    dp = [metrics[e]["mlp"]["aggregate"]["dp"]["wape"] * 100 for e in EPS]
    best_rc = [compute_best_rc(metrics, e) for e in EPS]
    x = np.arange(len(EPS))
    w = 0.28

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w, [nodp]*len(EPS), w, color=C["nodp"], label="nodp")
    ax.bar(x, dp, w, color=C["dp"], label="dp")
    ax.bar(x + w, best_rc, w, color=C["dp+rc"], label="dp+rc(best)")

    # 柱顶标注
    for i in range(len(EPS)):
        dp_cost = dp[i] - nodp
        rc_gain = best_rc[i] - dp[i]
        ax.text(x[i], dp[i] + 0.1, f"+{dp_cost:.1f}pp", ha="center", fontsize=8, color=C["dp"])
        ax.text(x[i]+w, best_rc[i] + 0.1, f"{rc_gain:+.1f}pp", ha="center", fontsize=8, color=C["dp+rc"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"ε={e}" for e in EPS])
    ax.set_ylabel("聚合 WAPE (%)")
    ax.set_title("三模式对比 + RC 增量消融（反归一化后）" if CN else "Three-mode comparison")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_rc_ablation.png", dpi=150)
    plt.close()
    print("图2 保存:", FIGS / "fig_rc_ablation.png")

# ============================================================================
# 图3 每客户端细粒度图（每档 ε 一张）
# ============================================================================
def fig3(metrics):
    for e in EPS:
        data = metrics[e]
        m = data["mlp"]["per_client"]
        best = compute_best_rc(metrics, e, per_client=True)
        cids = [c for c in CLIENT_ORDER if c in m]

        nodp_vals = [m[c]["nodp"]["wape"]*100 for c in cids]
        dp_vals = [m[c]["dp"]["wape"]*100 for c in cids]
        rc_vals = [best[c]["wape"]*100 if c in best else float("nan") for c in cids]

        x = np.arange(len(cids))
        w = 0.26
        fig, ax = plt.subplots(figsize=(13, 5.5))
        ax.bar(x - w, nodp_vals, w, color=C["nodp"], label="nodp")
        ax.bar(x, dp_vals, w, color=C["dp"], label="dp")
        ax.bar(x + w, rc_vals, w, color=C["dp+rc"], label="dp+rc(best)")

        # best 架构标注
        for i, c in enumerate(cids):
            if c in best and best[c]["arch"]:
                ax.text(x[i]+w, rc_vals[i]+0.3, best[c]["arch"], ha="center", fontsize=7, color="#333")

        ax.set_xticks(x)
        ax.set_xticklabels(cids, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("WAPE (%)")
        ax.set_title(f"每客户端细粒度对比 ε={e}（反归一化后）" if CN else f"Per-client ε={e}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        out = ANALYSIS / f"epsilon-{e}" / f"fig_per_client_epsilon_{e}.png"
        fig.savefig(out, dpi=150)
        plt.close()
        print(f"图3(ε={e}) 保存:", out)

# ============================================================================
# 图4 训练收敛曲线
# ============================================================================
def fig4():
    fig, ax = plt.subplots(figsize=(9, 6))
    # nodp
    with open(BASELINE / "nodp" / "checkpoints" / "baseline_history.json") as f:
        d = json.load(f)
    tl = d["train_losses"]
    ax.plot(tl, "o-", color=C["nodp"], ls="--", lw=1.5, label="nodp")
    # dp 各档
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(EPS)))
    for e, col in zip(EPS, colors):
        with open(BASELINE / "dp" / f"epsilon-{e}" / "checkpoints" / "baseline_history.json") as f:
            d = json.load(f)
        ax.plot(d["train_losses"], "o-", color=col, lw=1.2, label=f"dp ε={e}")
    ax.set_xlabel("联邦训练轮次")
    ax.set_ylabel("训练损失（归一化空间）")
    ax.set_title("训练收敛曲线" if CN else "Training convergence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.text(0.99, 0.01, "注：归一化空间损失（与反归一化图不同）", ha="right", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(FIGS / "fig_convergence.png", dpi=150)
    plt.close()
    print("图4 保存:", FIGS / "fig_convergence.png")

# ============================================================================
# 图5 隐私审计图（三面板）
# ============================================================================
def fig5():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 面板A: 参与率（叠加全部档）
    ax = axes[0]
    for e in EPS:
        with open(BASELINE / "dp" / f"epsilon-{e}" / "checkpoints" / "audit_log.json") as f:
            a = json.load(f)
        rounds = a["rounds"]
        joined = [len(r["joined"]) if isinstance(r["joined"], list) else r["joined"]
                  for r in rounds]
        expected = [len(r["expected"]) if isinstance(r["expected"], list) else r["expected"]
                    for r in rounds]
        ax.plot(joined, "-", lw=1, alpha=0.7, label=f"ε={e}")
    ax.set_xlabel("轮次")
    ax.set_ylabel("参与客户端数")
    ax.set_title("(A) 每轮参与客户端数")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 面板B: 每客户端隐私预算消耗（ε=0.5 示例）
    ax = axes[1]
    with open(BASELINE / "dp" / "epsilon-0.5" / "checkpoints" / "config.json") as f:
        c = json.load(f)
    pc = c["dp"]["per_client"]
    cids = [cid for cid in CLIENT_ORDER if cid in pc]
    eps_consumed = [pc[cid]["epsilon"] for cid in cids]
    ax.bar(range(len(cids)), eps_consumed, color="#66c2a5")
    ax.axhline(0.5, color="r", ls="--", lw=1, label="目标 ε=0.5")
    ax.set_xticks(range(len(cids)))
    ax.set_xticklabels(cids, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("实际消耗 ε")
    ax.set_title("(B) 每客户端隐私预算消耗")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # 面板C: 自适应裁剪范数
    ax = axes[2]
    for e in EPS:
        with open(BASELINE / "dp" / f"epsilon-{e}" / "checkpoints" / "audit_log.json") as f:
            a = json.load(f)
        rounds = a["rounds"]
        clip = [r.get("clip_norm") for r in rounds]
        ax.plot(clip, "-", lw=1.2, label=f"ε={e}")
    ax.set_xlabel("轮次")
    ax.set_ylabel("裁剪范数 C")
    ax.set_title("(C) 自适应裁剪范数")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle("隐私审计：参与率 / 预算消耗 / 裁剪范数")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_privacy_audit.png", dpi=150)
    plt.close()
    print("图5 保存:", FIGS / "fig_privacy_audit.png")

# ============================================================================
# 表6 汇总表（Markdown）
# ============================================================================
def table6(metrics):
    nodp = metrics[EPS[0]]["mlp"]["aggregate"]["nodp"]["wape"] * 100
    dp = [metrics[e]["mlp"]["aggregate"]["dp"]["wape"] * 100 for e in EPS]
    best_rc = [compute_best_rc(metrics, e) for e in EPS]

    lines = []
    lines.append("| 变体 | " + " | ".join([f"ε={e}" for e in EPS]) + " |")
    lines.append("|" + "---|" * (len(EPS) + 1))
    lines.append(f"| nodp | " + " | ".join([f"{nodp:.2f}%" for _ in EPS]) + " |")
    lines.append(f"| dp | " + " | ".join([f"{v:.2f}%" for v in dp]) + " |")
    lines.append(f"| dp+rc(best) | " + " | ".join([f"{v:.2f}%" for v in best_rc]) + " |")
    lines.append(f"| DP代价 (dp-nodp) | " + " | ".join([f"{d-nodp:+.2f}pp" for d in dp]) + " |")
    lines.append(f"| RC改进 (dp+rc-dp) | " + " | ".join([f"{r-d:+.2f}pp" for r, d in zip(best_rc, dp)]) + " |")

    md = "\n".join(lines)
    with open(FIGS / "summary_table.md", "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print("表6 保存:", FIGS / "summary_table.md")
    print(md)

# ============================================================================
# 图7 RC 架构消融图
# ============================================================================
def fig7(metrics):
    x = np.arange(len(EPS))
    w = 0.18
    arch_colors = {"mlp": "#ff9896", "lstm": "#aec7e8", "tcn": "#98df8a"}
    best_rc = [compute_best_rc(metrics, e) for e in EPS]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, a in enumerate(ARCHS):
        vals = [metrics[e][a]["aggregate"]["dp+rc"]["wape"]*100 for e in EPS]
        ax.bar(x + (i-1)*w, vals, w, color=arch_colors[a], label=f"{a}")
    ax.bar(x + w, best_rc, w, color=C["dp+rc"], edgecolor="black", label="best (自选)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"ε={e}" for e in EPS])
    ax.set_ylabel("聚合 WAPE (%)")
    ax.set_title("RC 架构消融：固定架构 vs 每客户端自选 best" if CN else "RC architecture ablation")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_rc_arch_ablation.png", dpi=150)
    plt.close()
    print("图7 保存:", FIGS / "fig_rc_arch_ablation.png")

# ============================================================================
def main():
    metrics = load_all_metrics()
    print(f"已加载 {sum(len(v) for v in metrics.values())} 份 JSON\n")
    fig1(metrics)
    fig2(metrics)
    fig3(metrics)
    fig4()
    fig5()
    table6(metrics)
    fig7(metrics)
    print("\n=== 全部图集生成完成 ===")

if __name__ == "__main__":
    main()
