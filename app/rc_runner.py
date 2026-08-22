"""二阶段：下载全局模型 → 本地 RC 训练（train_personalized）→ 上传指标+对比图。"""
from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def download_model_bytes(server_url: str, token: str, task_id: int) -> bytes:
    """下载最终模型（pickle: {keys, tensors}）。"""
    import urllib.request
    req = urllib.request.Request(
        server_url + f"/api/tasks/{task_id}/model",
        headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


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
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status == 200
