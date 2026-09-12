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
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 单仓库多客户端：阶段状态/工作目录按进程隔离（FL_STAGE_DIR / FL_RC_WORK 由 agent 启动时设置）
STAGE_DIR = Path(os.environ.get("FL_STAGE_DIR",
                                str(Path(__file__).resolve().parent / "stage")))
WORK_DIR = Path(os.environ.get("FL_RC_WORK",
                               str(Path(__file__).resolve().parent / "rc_work")))

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
    """供 /local/stage-status 调用。

    必须带上 ``error``：前端在 stage2/stage3 为 failed 时会展示它
    （views.js 读 st.error）。早期版本漏了这个字段，界面就永远只显示
    "未知错误，查看 agent 窗口日志"，把真正的失败原因（模型未就绪 /
    未采集数据 / 身份为空）全藏住了。
    """
    s = _read(task_id)
    out = {
        "task_id": task_id,
        "stage2": s.get("stage2", "idle"),
        "stage3": s.get("stage3", "idle"),
        "stage2_epoch_losses": s.get("epoch_losses", []),
        "corrector_arch": s.get("corrector_arch"),
        "wape_global": s.get("wape_global"),
        "wape_rc": s.get("wape_rc"),
    }
    if s.get("error"):
        out["error"] = s["error"]
    return out


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


def _child_env() -> dict:
    """让子进程固定以 UTF-8 输出（父进程也按 UTF-8 解码）。

    为什么必须显式指定：Windows 下 subprocess 的 text=True 默认按
    locale(GBK/cp936) 解码。一旦子进程输出 UTF-8 字节（例如外层环境设了
    PYTHONIOENCODING=utf-8，IDE/终端很常见），reader 线程会抛
    UnicodeDecodeError，导致 res.stdout 变成 None，随后解析直接崩
    （实测复现：AttributeError: 'NoneType' object has no attribute 'splitlines'）。
    这里两端都钉死 UTF-8，与外部环境无关。
    """
    return {**os.environ, "PYTHONIOENCODING": "utf-8"}


def _read_client_metrics(result_json: Path, client_id: str) -> dict:
    """从 train_personalized 产出的 personalized_results.json 取本客户端结果。

    ⚠️ 键名陷阱：train_personalized 把每客户端结果放在顶层 ``results`` 下
    （以 client_id 为键）。本模块早期误读 ``client_metrics``，导致 cid_data
    恒为空、``wape_global`` / ``wape_rc`` 永远写成 None。
    这里优先 ``results``，并保留 ``client_metrics`` 兼容。
    """
    if not result_json.exists():
        return {}
    try:
        data = json.loads(result_json.read_text(encoding="utf-8"))
    except Exception:
        return {}
    for key in ("results", "client_metrics"):
        per = (data.get(key) or {}).get(client_id)
        if per:
            return per
    return {}


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
        # 前置校验：二阶段依赖"已加入任务的客户端身份 + 已采集的本地数据 +
        # 一阶段产出的全局模型"三件事，任缺其一都在子进程里炸得很难看
        # （实测只看到 HTTP 404 / KeyError），这里提前给出可操作的提示。
        if not client_id:
            raise RuntimeError(
                "本客户端身份为空，无法开启二阶段。请先在平台「加入任务」"
                "并完成数据采集，确认客户端身份已绑定后重试。")
        _ddir = (Path(data_dir) if data_dir
                 else Path(__file__).resolve().parent / "data")
        if not _ddir.exists() or not list(_ddir.glob("*.csv")):
            raise RuntimeError(
                f"未找到已采集的数据（目录：{_ddir}）。"
                f"请先在本客户端点击「采集数据」下载数据源，再开启二阶段。")

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
                             encoding="utf-8", errors="replace",
                             env=_child_env(), timeout=1800)

        # 解析 personalized_results.json（train_personalized 已含 per-client 结果）
        result_json = out_root / rc_type / "personalized_results.json"
        cid_data = _read_client_metrics(result_json, client_id)

        if res.returncode != 0:
            raise RuntimeError(f"二阶段训练失败: {(res.stderr or '')[-800:]}")

        # 从结果里取 epoch_losses / wape（若 train_personalized 该分支未存则留空）
        epoch_losses = cid_data.get("epoch_losses", [])
        if not epoch_losses:
            # 尝试从子进程打印行解析 "Epoch k/n  loss=x.xxxxxx"
            epoch_losses = _parse_epoch_losses(res.stdout)

        state["stage2"] = "done"
        state.pop("error", None)   # 清掉上一次失败遗留的错误，避免界面误报
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


def _parse_epoch_losses(stdout: str | None) -> list[float]:
    out: list[float] = []
    text = stdout or ""      # 抓取失败时 stdout 可能为 None，不能直接 splitlines
    # 优先取完整逐轮行 EPOCHLOSS k loss（train_personalized 已加）
    for line in text.splitlines():
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
    for line in text.splitlines():
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
        if not client_id:
            raise RuntimeError(
                "本客户端身份为空，无法生成本地预测。请先加入任务并完成采集。")
        _ddir = (Path(data_dir) if data_dir
                 else Path(__file__).resolve().parent / "data")
        if not _ddir.exists() or not list(_ddir.glob("*.csv")):
            raise RuntimeError(
                f"未找到已采集的数据（目录：{_ddir}）。请先采集数据再生成预测。")

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
                             encoding="utf-8", errors="replace",
                             env=_child_env(), timeout=1800)
        if res.returncode != 0 or not out_json.exists():
            raise RuntimeError(f"预测生成失败: {(res.stderr or '')[-800:]}")

        state["stage3"] = "ready"
        state.pop("error", None)   # 同上：成功即清掉遗留错误
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
