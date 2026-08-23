"""5 档隐私预算全量流水线驱动脚本。

对 ε = [7.5, 5.5, 3.5, 1.5, 0.5] 依次执行：
  0. nodp 基线滚动评估一次 → fl_code/analysis/nodp_denorm_metrics.json
  1. DP 联邦训练（train_baseline，30 轮，自适应裁剪 + clip-count-noise 5.0）
     → fl_code/baseline_outputs/dp/epsilon-<ε>/
  2. 3 种 RC 训练（train_personalized：mlp / lstm / tcn）
     → fl_code/personalized_outputs/epsilon-<ε>/<arch>/
  3. 反归一化验证（validate_denorm，nodp 指标复用步骤 0 的结果，
     只评估 dp / dp+rc 变体）→ fl_code/analysis/epsilon-<ε>/denorm_metrics_<arch>.json
全部 5 档完成后：
  4. 标准图集（make_figures）→ fl_code/analysis/figs/ + summary_table.md

用法：
  python -m fl_code.run_all_epsilon --force     # 删除旧产物全量重跑
  python -m fl_code.run_all_epsilon             # 跳过已完成项（断点续跑）
  python -m fl_code.run_all_epsilon --smoke     # 冒烟验证：临时目录、小规模、不碰正式产物
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "fl_code" / "baseline_outputs"
PERSONALIZED_DIR = ROOT / "fl_code" / "personalized_outputs"
ANALYSIS_DIR = ROOT / "fl_code" / "analysis"
LOG_DIR = ROOT / "fl_code" / "logs"

EPS = [7.5, 5.5, 3.5, 1.5, 0.5]
ARCHS = ["mlp", "lstm", "tcn"]
DEFAULT_CLIENTS = ["steel_ind_0", "tetouan_city_0", "tetouan_city_1",
                   "tetouan_city_2", "lcl_res_0", "lcl_res_1",
                   "eld_ind_0", "eld_ind_1", "eld_ind_2"]
SMOKE_CLIENTS = ["steel_ind_0", "lcl_res_0"]
# nodp 对照模型（旧实验产物，不重跑）；validate_denorm 自动解析只会找
# baseline_outputs/nodp（无连字符），实际目录是 no-dp，故必须显式传
NODP_GLOBAL = BASELINE_DIR / "no-dp" / "checkpoints" / "round_030.pt"


def _fmt(e: float) -> str:
    return f"{e:g}"


def run_step(name: str, cmd: list[str], log_path: Path) -> bool:
    """Run one subprocess, teeing output to console + log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ===\n$ {' '.join(cmd)}")
    started = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", cwd=ROOT)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            print(line, end="", flush=True)
        proc.wait()
    ok = proc.returncode == 0
    print(f"[{'OK' if ok else 'FAIL'}] {name} "
          f"({time.perf_counter() - started:.0f}s, exit={proc.returncode}) "
          f"log: {log_path}")
    return ok


def _python() -> list[str]:
    return [sys.executable, "-u", "-m"]


# ---------------------------------------------------------------------------
# 完成判定
# ---------------------------------------------------------------------------

def baseline_done(eps_dir: Path, rounds: int) -> bool:
    return (eps_dir / "baseline_history.json").exists() and (
        eps_dir / "checkpoints" / f"round_{rounds:03d}.pt").exists()


def rc_done(arch_dir: Path, n_clients: int) -> bool:
    if not (arch_dir / "config.json").exists():
        return False
    correctors = list(arch_dir.glob("corrector_*.pt"))
    return len(correctors) >= n_clients


def validate_done(e: float, arch: str, analysis_root: Path) -> bool:
    return (analysis_root / f"epsilon-{_fmt(e)}"
            / f"denorm_metrics_{arch}.json").exists()


# ---------------------------------------------------------------------------
# 各步骤
# ---------------------------------------------------------------------------

def step_nodp(args, analysis_root: Path) -> bool:
    """nodp 基线滚动评估，仅一次；其余验证步骤通过 --nodp-json 复用结果。"""
    out = analysis_root / "nodp_denorm_metrics.json"
    if not args.force and out.exists():
        print("[skip] nodp 基线评估 已完成")
        return True
    cmd = _python() + ["fl_code.validate_denorm",
                       "--nodp-global", str(NODP_GLOBAL),
                       "--output-json", str(out),
                       "--fig-dir", str(analysis_root)]
    if args.clients:
        cmd += ["--clients", *args.clients]
    if args.smoke:
        cmd += ["--max-seqs", "1", "--eval-seqs", "1"]
    ok = run_step("nodp 基线评估（仅一次）", cmd,
                  args.log_dir / "nodp_eval.log")
    return ok

def step_baseline(args, e: float, eps_dir: Path) -> bool:
    if not args.force and baseline_done(eps_dir, args.rounds):
        print(f"[skip] baseline ε={_fmt(e)} 已完成")
        return True
    if args.force and eps_dir.exists():
        shutil.rmtree(eps_dir)
    cmd = _python() + ["fl_code.train_baseline",
                       "--dp-epsilon", _fmt(e),
                       "--dp-adaptive-clip",
                       "--dp-clip-count-noise", "5.0",
                       "--rounds", str(args.rounds),
                       "--output-dir", str(eps_dir)]
    if args.clients:
        cmd += ["--clients", *args.clients]
    if args.smoke:
        cmd += ["--max-seqs", "2", "--eval-seqs", "1"]
    ok = run_step(f"baseline ε={_fmt(e)}", cmd,
                  args.log_dir / f"baseline_eps{_fmt(e)}.log")
    if ok:
        # train_baseline 把产物写到 <output-dir>/dp/，上移一层与现状目录一致
        nested = eps_dir / "dp"
        if nested.is_dir():
            for entry in nested.iterdir():
                shutil.move(str(entry), str(eps_dir / entry.name))
            nested.rmdir()
    return ok


def step_rc(args, e: float, arch: str, eps_out: Path,
            global_path: Path) -> bool:
    arch_dir = eps_out / arch
    if not args.force and rc_done(arch_dir, len(args.clients or DEFAULT_CLIENTS)):
        print(f"[skip] RC ε={_fmt(e)} {arch} 已完成")
        return True
    if args.force and arch_dir.exists():
        shutil.rmtree(arch_dir)
    cmd = _python() + ["fl_code.train_personalized",
                       "--rc-type", arch,
                       "--global-model", str(global_path),
                       "--output-dir", str(eps_out)]
    if args.clients:
        cmd += ["--clients", *args.clients]
    if args.smoke:
        cmd += ["--epochs", "2", "--max-seqs", "1", "--eval-seqs", "1"]
    ok = run_step(f"RC ε={_fmt(e)} {arch}", cmd,
                  args.log_dir / f"rc_eps{_fmt(e)}_{arch}.log")
    return ok


def step_validate(args, e: float, arch: str, global_path: Path,
                  eps_out: Path, analysis_root: Path) -> bool:
    if not args.force and validate_done(e, arch, analysis_root):
        print(f"[skip] 验证 ε={_fmt(e)} {arch} 已完成")
        return True
    out_json = (analysis_root / f"epsilon-{_fmt(e)}"
                / f"denorm_metrics_{arch}.json")
    fig_dir = analysis_root / f"epsilon-{_fmt(e)}"
    cmd = _python() + ["fl_code.validate_denorm",
                       "--nodp-json", str(analysis_root / "nodp_denorm_metrics.json"),
                       "--dp-global", str(global_path),
                       "--rc-dir", str(eps_out / arch),
                       "--rc-type", arch,
                       "--output-json", str(out_json),
                       "--fig-dir", str(fig_dir)]
    if args.clients:
        cmd += ["--clients", *args.clients]
    if args.smoke:
        cmd += ["--max-seqs", "1", "--eval-seqs", "1"]
    ok = run_step(f"验证 ε={_fmt(e)} {arch}", cmd,
                  args.log_dir / f"validate_eps{_fmt(e)}_{arch}.log")
    return ok


def step_figures(args) -> bool:
    if args.smoke:
        return True
    cmd = _python() + ["fl_code.make_figures"]
    return run_step("可视化 make_figures", cmd,
                    args.log_dir / "make_figures.log")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="删除已完成目标目录并全量重跑（默认：跳过已完成项）")
    parser.add_argument("--smoke", action="store_true",
                        help="冒烟模式：临时目录 + 小规模，不写正式产物目录")
    parser.add_argument("--rounds", type=int, default=30,
                        help="baseline 联邦训练轮数（默认 30；冒烟自动 1）")
    parser.add_argument("--clients", nargs="*", default=None,
                        help="参与客户端（默认全部 9 个）")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if args.smoke:
        args.rounds = 1
        args.clients = SMOKE_CLIENTS

    if args.smoke:
        tmp = Path(tempfile.mkdtemp(prefix="fl_eps_smoke_"))
        baseline_root, rc_root, analysis_root = tmp / "baseline", tmp / "rc", tmp / "analysis"
        args.log_dir = tmp / "logs"
        print(f"SMOKE 模式：全部产物写入临时目录 {tmp}")
    else:
        baseline_root, rc_root, analysis_root = (
            BASELINE_DIR, PERSONALIZED_DIR, ANALYSIS_DIR)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_dir = LOG_DIR / f"run_all_epsilon_{ts}"
    args.log_dir.mkdir(parents=True, exist_ok=True)

    total = 1 + len(EPS) * (1 + len(ARCHS) * 2) + (0 if args.smoke else 1)
    done, failed = 0, []

    nodp_ok = step_nodp(args, analysis_root)
    done += 1
    if not nodp_ok:
        failed.append("nodp 基线评估")

    for e in EPS:
        eps_dir = baseline_root / "dp" / f"epsilon-{_fmt(e)}"
        # baseline
        ok = step_baseline(args, e, eps_dir)
        done += 1
        if not ok:
            failed.append(f"baseline ε={_fmt(e)}")
            continue
        global_path = eps_dir / "checkpoints" / f"round_{args.rounds:03d}.pt"
        eps_out = rc_root / f"epsilon-{_fmt(e)}"
        for arch in ARCHS:
            if not step_rc(args, e, arch, eps_out, global_path):
                failed.append(f"RC ε={_fmt(e)} {arch}")
            elif nodp_ok:
                if not step_validate(args, e, arch, global_path,
                                     eps_out, analysis_root):
                    failed.append(f"验证 ε={_fmt(e)} {arch}")
            else:
                failed.append(f"验证 ε={_fmt(e)} {arch}（nodp 评估失败，跳过）")
            done += 2

    ok_figures = step_figures(args)
    done += 1
    if not ok_figures:
        failed.append("可视化 make_figures")

    print(f"\n{'=' * 60}\n完成 {done}/{total} 步，失败 {len(failed)} 步"
          f"（{'冒烟' if args.smoke else '正式'} 模式）")
    if failed:
        print("失败清单：")
        for f in failed:
            print(f"  - {f}")
        if args.smoke:
            print(f"冒烟产物保留于 {baseline_root.parent}（检查后自行删除）")
        return 1
    print(f"全部成功。日志：{args.log_dir}")
    if args.smoke:
        print(f"冒烟产物保留于 {baseline_root.parent}（检查后自行删除）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
