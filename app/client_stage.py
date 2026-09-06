"""客户端二阶段/预测本地编排（need.md 阶段2/3）。

本模块只负责"本地编排 + 状态记录"，真正的二阶段训练/滚动预测
复用 fl_code 已跑通的 train_personalized / app_stage_eval。

本地状态按任务存到 app/stage/<task_id>.json：
  { stage2: "idle"|"running"|"done"|"failed",
    epoch_losses: [..],          # 阶段2 修正器逐轮 loss
    corrector_arch: "tcn"|..,    # 该客户端使用的修正器架构
    stage3: "idle"|"ready",
    stage3_data_path: "...",     # 阶段3 预测数据 JSON
    wape_global, wape_rc, wape_drop_pct, pinaw, coverage,
    started_at, finished_at }
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE_DIR = Path(__file__).resolve().parent / "stage"
WORK_DIR = Path(__file__).resolve().parent / "rc_work"   # 复用 rc 的临时工作目录

RC_TYPE_DEFAULT = "tcn"          # 客户端修正器架构（与 server 端一致时按配置）

# ---------------------------------------------------------------------------
# 状态读写
# ---------------------------------------------------------------------------


def _state_path(task_id: int) -> Path:
    return STAGE_DIR / f"task{task_id}.json"


def _read(task_id: int) -> dict:
    p = _state_path(task_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _write(task_id: int, state: dict) -> None:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    (_state_path(task_id)).write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def get_status(task_id: int) -> dict:
    """供 /local/stage-status 调用。"""
    s = _read(task_id)
    return {
        "task_id": task_id,
        "stage2": s.get("stage2", "idle"),
        "stage3": s.get("stage3", "idle"),
        "stage2_epoch_losses": s.get("epoch_losses", []),
        "corrector_arch": s.get("corrector_arch"),
    }


# ---------------------------------------------------------------------------
# 阶段2：本地训练残差修正器（复用 train_personalized 单客户端跑一遍）
# ---------------------------------------------------------------------------

def _download_global_model(server_url: str, token: str, task_id: int) -> Path:
    """下载 server 全局模型并落成 .pt state_dict（复用 rc_runner）。"""
    from app.rc_runner import download_model_bytes, parse_model_bytes
    raw = download_model_bytes(server_url, token, task_id)
    keys, tensors = parse_model_bytes(raw)
    import torch
    from fl_code.fed_core.params import tensors_to_state_dict
    from fl_code.models import TCNConfig, build_tcn
    state = tensors_to_state_dict(tensors, keys)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    pt = WORK_DIR / f"global_task{task_id}.pt"
    torch.save(state, pt)
    return pt


def run_stage2(server_url: str, token: str, task_id: int,
               client_id: str, rc_type: str = RC_TYPE_DEFAULT,
               data_dir: str | None = None) -> dict:
    """阶段2：下载全局模型 → 本地训练修正器 → 记录 epoch_losses。

    复用 rc_runner 已跑通的子进程调用 train_personalized。
    """
    state = _read(task_id)
    state["stage2"] = "running"
    state["started_at"] = _now()
    _write(task_id, state)

    try:
        global_pt = _download_global_model(server_url, token, task_id)
        out_root = WORK_DIR / f"stage2_task{task_id}"
        out_root.mkdir(parents=True, exist_ok=True)

        cmd = [sys.executable, "-m", "fl_code.train_personalized",
               "--global-model", str(global_pt),
               "--output-dir", str(out_root),
               "--clients", client_id,
               "--epochs", "15",
               "--rc-type", rc_type]
        if data_dir:
            cmd += ["--data-dir", data_dir]
        # 输出到临时目录，避免写正式产物目录
        res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                             timeout=1800)

        # 解析 personalized_results.json（train_personalized 已含 per-client 结果）
        result_json = out_root / rc_type / "personalized_results.json"
        cid_data = {}
        if result_json.exists():
            try:
                all_res = json.loads(result_json.read_text(encoding="utf-8"))
                cid_data = all_res.get("client_metrics", {}).get(client_id, {}) or {}
                # 兼容：某些版本把 per-client 结果放 client_metrics
                cid_data = all_res.get("client_metrics", {}).get(client_id, {}) or {}
            except Exception:
                cid_data = {}

        if res.returncode != 0:
            raise RuntimeError(f"二阶段训练失败: {res.stderr[-800:]}")

        # 从结果里取 epoch_losses / wape（若 train_personalized 该分支未存则留空）
        epoch_losses = cid_data.get("epoch_losses", [])
        if not epoch_losses:
            # 尝试从子进程打印行解析 "Epoch k/n  loss=x.xxxxxx"
            epoch_losses = _parse_epoch_losses(res.stdout)

        state["stage2"] = "done"
        state["epoch_losses"] = epoch_losses
        state["corrector_arch"] = rc_type
        state["wape_global"] = cid_data.get("wape_baseline")
        state["wape_rc"] = cid_data.get("wape_personalized")
        state["finished_at"] = _now()
        _write(task_id, state)
        return state
    except Exception as e:
        state["stage2"] = "failed"
        state["error"] = str(e)
        _write(task_id, state)
        raise


def _parse_epoch_losses(stdout: str) -> list[float]:
    out: list[float] = []
    # 优先取完整逐轮行 EPOCHLOSS k loss（train_personalized 已加）
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("EPOCHLOSS"):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    out.append(round(float(parts[2]), 6))
                except Exception:
                    continue
    if out:
        return out
    # 兜底：解析 "Epoch k/n  loss=x.xxxxxx"
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("Epoch ") and "loss=" in line:
            try:
                val = float(line.split("loss=")[1].split()[0])
                out.append(round(val, 6))
            except Exception:
                continue
    return out


def _now() -> str:
    from datetime import datetime
    return str(datetime.utcnow())


# ---------------------------------------------------------------------------
# 阶段3：本地测试集预测 → 生成展示数据（交给 app_stage_eval 子进程）
# ---------------------------------------------------------------------------

def run_stage3(server_url: str, token: str, task_id: int,
               client_id: str, rc_type: str = RC_TYPE_DEFAULT,
               data_dir: str | None = None) -> dict:
    """阶段3：下载全局模型 + 加载阶段2保存的修正器 → 滚动预测 → 写 stage3 数据。"""
    state = _read(task_id)
    if state.get("stage2") != "done":
        raise RuntimeError("请先完成二阶段训练（开启二阶段）")

    try:
        global_pt = _download_global_model(server_url, token, task_id)
        corr_pt = WORK_DIR / f"stage2_task{task_id}" / rc_type / f"corrector_{client_id}.pt"
        if not corr_pt.exists():
            raise RuntimeError("未找到本地修正器模型，请重新跑二阶段")

        out_json = WORK_DIR / f"stage3_task{task_id}_{client_id}.json"
        cmd = [sys.executable, "-m", "fl_code.app_stage_eval",
               "--global-model", str(global_pt),
               "--corrector", str(corr_pt),
               "--rc-type", rc_type,
               "--cid", client_id,
               "--out", str(out_json)]
        if data_dir:
            cmd += ["--data-dir", data_dir]
        res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                             timeout=1800)
        if res.returncode != 0 or not out_json.exists():
            raise RuntimeError(f"预测生成失败: {res.stderr[-800:]}")

        state["stage3"] = "ready"
        state["stage3_data_path"] = str(out_json)
        _write(task_id, state)
        return state
    except Exception as e:
        state["stage3"] = "failed"
        state["error"] = str(e)
        _write(task_id, state)
        raise


def get_stage3_data(task_id: int) -> dict:
    """供 /local/stage3-data 调用；数据未就绪抛异常。"""
    s = _read(task_id)
    p = s.get("stage3_data_path")
    if s.get("stage3") != "ready" or not p or not Path(p).exists():
        raise FileNotFoundError("预测数据未就绪")
    return json.loads(Path(p).read_text(encoding="utf-8"))
