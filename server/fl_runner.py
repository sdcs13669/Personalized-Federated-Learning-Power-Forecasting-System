"""Manage flwr server lifecycle via subprocess workers.

flwr 1.30 `start_server` 在后台线程里无法正常监听 gRPC 端口，
改为在独立子进程（fl_server_worker.py）中运行。
"""
from __future__ import annotations

import logging
import os
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/data"))


@dataclass
class ActiveTask:
    task_id: int
    proc: subprocess.Popen | None = None
    state_keys: list[str] = field(default_factory=list)


_active_tasks: dict[int, ActiveTask] = {}


def _model_path(task_id: int) -> Path:
    return MODEL_DIR / f"fl_model_{task_id}.pkl"


def get_final_model(task_id: int) -> bytes | None:
    """Get final model parameters as bytes (worker saves to file)."""
    p = _model_path(task_id)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return f.read()


def get_task_status(task_id: int) -> str | None:
    """Check if a task is actively training (based on subprocess)."""
    active = _active_tasks.get(task_id)
    if active is None or active.proc is None:
        return None
    if active.proc.poll() is None:
        return "training"
    return "completed" if active.proc.returncode == 0 else "failed"


def start_training(task_dict: dict, participants: list[dict],
                   db_session_factory) -> None:
    """Start flwr server in a separate subprocess. Non-blocking."""
    task_id = task_dict["id"]

    # Guard against port collision (demo: only one concurrent training task)
    for tid, active in _active_tasks.items():
        if active.proc is not None and active.proc.poll() is None:
            raise RuntimeError(
                f"Task {tid} is already training. "
                f"Only one concurrent task supported (gRPC port collision)."
            )

    payload = {**task_dict, "participants": participants}
    task_pkl = Path(tempfile.gettempdir()) / f"fl_task_{task_id}.pkl"
    with open(task_pkl, "wb") as f:
        pickle.dump(payload, f)

    model_out = _model_path(task_id)
    if model_out.exists():
        model_out.unlink()

    worker = ROOT / "server" / "fl_server_worker.py"
    log_path = MODEL_DIR / f"fl_worker_{task_id}.log"
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            [sys.executable, str(worker), str(task_pkl), str(model_out)],
            cwd=str(ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    _active_tasks[task_id] = ActiveTask(task_id=task_id, proc=proc)
    logger.info("Started flwr server subprocess for task %d (log: %s)",
                task_id, log_path)


def cleanup_completed_tasks(max_age_hours: int = 24) -> int:
    """Demo stub — keep completed tasks so get_final_model() still works."""
    return 0
