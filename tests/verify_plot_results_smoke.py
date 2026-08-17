"""End-to-end smoke verification for the 3-stage plot_results rework.

Covers:
  V1  unit tests for _stage_table (tests/test_plot_results.py)
  V2  train_personalized smoke (tiny) -> personalized_results.json carries
      finite wape_baseline / wape_personalized
  V3  compare-figure structure built from the fresh JSON: dp+rc WAPE is
      picked up (no "re-run" annotation), 6 bars per client, 6 legend entries
  V4  plot_results CLI smoke -> 4 PNGs in a temp fig dir

All outputs go to C:/tmp/verify_plot_<timestamp> — never touches
fl_code/baseline_outputs, personalized_outputs or figures.

Usage:  python tests/verify_plot_results_smoke.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TMP = Path("C:/tmp") / f"verify_plot_{time.strftime('%Y%m%d_%H%M%S')}"
BASELINE_ROOT = ROOT / "fl_code" / "baseline_outputs"


def _newest_nodp_ckpt() -> Path:
    ckpts = sorted((BASELINE_ROOT / "nodp" / "checkpoints").glob("round_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not ckpts:
        raise SystemExit("No nodp checkpoints — run train_baseline first")
    return ckpts[-1]


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def v1_unit_tests() -> None:
    _run([sys.executable, "-m", "pytest", "tests/test_plot_results.py", "-q"])
    print("[V1 OK] unit tests pass\n")


def v2_personalized_wape(pers_out: Path) -> dict:
    _run([sys.executable, "-m", "fl_code.train_personalized",
          "--global-model", str(_newest_nodp_ckpt()),
          "--clients", "steel_ind_0",
          "--max-seqs", "5", "--eval-seqs", "2", "--epochs", "1",
          "--output-dir", str(pers_out)])
    # 输出按 rc 类型分文件夹（默认 mlp）
    with open(pers_out / "mlp" / "personalized_results.json") as f:
        res = json.load(f)
    fm = res["final_metrics"]
    r = res["results"]["steel_ind_0"]
    for key, value in (
        ("final_metrics.wape_personalized", fm["wape_personalized"]),
        ("final_metrics.wape_baseline", fm["wape_baseline"]),
        ("results.wape_personalized", r["wape_personalized"]),
        ("results.wape_baseline", r["wape_baseline"]),
    ):
        if not np.isfinite(value):
            raise SystemExit(f"V2 FAILED: {key} = {value}")
        print(f"  {key} = {value:.4f}")
    print("[V2 OK] personalized smoke writes finite WAPE\n")
    return res


def v3_compare_figure(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fl_code.plot_results import _fig_compare, _labels, _stage_table

    with open(BASELINE_ROOT / "nodp" / "baseline_history.json") as f:
        nodp = json.load(f)
    with open(BASELINE_ROOT / "dp" / "baseline_history.json") as f:
        dp = json.load(f)

    table = _stage_table(nodp, dp, res)
    if not np.isfinite(table["dp+rc"]["wape"]):
        raise SystemExit(f"V3 FAILED: dp+rc wape = {table['dp+rc']['wape']}")
    fig = _fig_compare(table, _labels(False), plt)
    ax_left, ax_right = fig.axes
    n_clients = len({c for s in ("nodp", "dp", "dp+rc")
                     for c in table[s]["clients"]})
    if len(ax_left.patches) != 6:
        raise SystemExit("V3 FAILED: left panel should have 6 bars")
    if len(ax_right.patches) != 6 * n_clients:
        raise SystemExit(f"V3 FAILED: right panel bars = {len(ax_right.patches)}")
    if len(ax_right.get_legend().get_texts()) != 6:
        raise SystemExit("V3 FAILED: legend should have 6 entries")
    if ax_left.texts:
        raise SystemExit("V3 FAILED: unexpected 're-run Phase 3' annotation")
    plt.close(fig)
    print(f"[V3 OK] compare figure: dp+rc WAPE={table['dp+rc']['wape']:.4f}, "
          f"{n_clients} clients x 6 bars, no annotation\n")


def v4_plot_results_cli(figs_out: Path, pers_json: Path) -> None:
    _run([sys.executable, "-m", "fl_code.plot_results",
          "--fig-dir", str(figs_out),
          "--personalized-json", str(pers_json)])
    for name in ("fig_phase1_nodp.png", "fig_phase2_dp.png",
                 "fig_phase3_personalized.png", "fig_phase_compare.png"):
        if not (figs_out / name).exists():
            raise SystemExit(f"V4 FAILED: missing {name}")
        print(f"  {figs_out / name}")
    print("[V4 OK] plot_results writes 4 figures\n")


def main() -> None:
    pers_out = TMP / "pers"
    figs_out = TMP / "figs"
    pers_out.mkdir(parents=True, exist_ok=True)
    print(f"Temp root: {TMP}\n")

    v1_unit_tests()
    res = v2_personalized_wape(pers_out)
    v3_compare_figure(res)
    v4_plot_results_cli(figs_out, pers_out / "mlp" / "personalized_results.json")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
