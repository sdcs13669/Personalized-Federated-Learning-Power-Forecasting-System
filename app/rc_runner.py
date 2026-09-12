"""二阶段：下载全局模型 → 本地 RC 训练（train_personalized）→ 上传指标+对比图。"""
from __future__ import annotations

import pickle
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 访问服务端（Tailscale/内网地址）必须强制直连：系统代理（Clash 等）会劫持导致
# 二阶段“下载全局模型” 502 Bad Gateway / 超时。数据集下载（collector，走 GitHub）
# 仍使用默认 opener 以便复用系统代理。
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def download_model_bytes(server_url: str, token: str, task_id: int) -> bytes:
    """下载最终模型（pickle: {keys, tensors}）。

    失败时必须给出可操作的提示：二阶段/预测展示的第一步就是这个下载，
    而服务端在"一阶段未完成或被停止"时只会回 404，原生异常
    （HTTP Error 404: Not Found）对使用者毫无线索，实测就是这么卡住的。
    """
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        server_url + f"/api/tasks/{task_id}/model",
        headers={"Authorization": "Bearer " + token})
    try:
        with _NO_PROXY_OPENER.open(req, timeout=30) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(
                f"任务 {task_id} 的全局模型尚未生成，无法开始二阶段。"
                f"常见原因：一阶段联邦训练还没跑完，或任务被「强制停止」"
                f"（停止的任务不会保存模型）。请确认任务状态为「已完成」后重试。"
            ) from e
        if e.code in (401, 403):
            raise RuntimeError(
                f"下载全局模型被拒绝（HTTP {e.code}）：请确认本客户端已登录平台，"
                f"并且已经加入任务 {task_id}。") from e
        raise RuntimeError(
            f"下载全局模型失败（HTTP {e.code}）：{e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"无法连接服务端 {server_url}：{e.reason}。"
            f"请确认服务端容器在线、网络（Tailscale）连通。") from e


def parse_model_bytes(raw: bytes) -> tuple[list[str], list]:
    data = pickle.loads(raw)
    return data["keys"], data["tensors"]


def run_rc(trainer_cmd: list[str], cwd: Path) -> None:
    """subprocess 调用 train_personalized（输出到临时目录）。"""
    subprocess.run(trainer_cmd, cwd=str(cwd), check=True)


def upload_rc_result(server_url: str, token: str, task_id: int,
                     client_id: str, wape_global: float, wape_rc: float,
                     png_path: str | None) -> bool:
    """上传 RC 指标与对比图到 server（multipart/form-data）。"""
    import urllib.request
    boundary = "----rcboundary"
    parts = []
    for name, val in [("client_id", client_id),
                      ("wape_global", str(wape_global)),
                      ("wape_rc", str(wape_rc))]:
        parts.append(
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{name}\"\r\n\r\n"
            f"{val}\r\n")
    if png_path and Path(png_path).exists():
        img = Path(png_path).read_bytes()
        parts.append(
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"png\"; "
            f"filename=\"cmp.png\"\r\nContent-Type: image/png\r\n\r\n")
        parts.append(img.decode("latin1"))
        parts.append("\r\n")
    parts.append(f"--{boundary}--\r\n")
    body = "".join(parts).encode("latin1")
    req = urllib.request.Request(
        server_url + f"/api/tasks/{task_id}/rc-result",
        data=body, method="POST",
        headers={"Authorization": "Bearer " + token,
                 "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with _NO_PROXY_OPENER.open(req, timeout=30) as resp:
        return resp.status == 200
