"""Manage flwr server lifecycle in background threads."""
from __future__ import annotations

import json
import logging
import pickle
import threading
from dataclasses import dataclass, field
from datetime import datetime

from server.models import Task, AuditRound

logger = logging.getLogger(__name__)


@dataclass
class ActiveTask:
    task_id: int
    thread: threading.Thread | None = None
    stop_event: threading.Event | None = None
    final_tensors: list | None = None
    state_keys: list[str] = field(default_factory=list)


_active_tasks: dict[int, ActiveTask] = {}


def get_final_model(task_id: int) -> bytes | None:
    """Get final model parameters as bytes (from memory, never from disk)."""
    active = _active_tasks.get(task_id)
    if active is None or active.final_tensors is None:
        return None
    return pickle.dumps({
        "keys": active.state_keys,
        "tensors": active.final_tensors,
    })


def get_task_status(task_id: int) -> str | None:
    """Check if a task is actively training."""
    active = _active_tasks.get(task_id)
    if active is None:
        return None
    if active.thread is not None and active.thread.is_alive():
        return "training"
    if active.final_tensors is not None:
        return "completed"
    return "unknown"


def start_training(task_dict: dict, participants: list[dict],
                   db_session_factory) -> None:
    """Start flwr server in a background thread. Non-blocking."""
    task_id = task_dict["id"]

    # Guard against port collision (demo: only one concurrent training task)
    for tid, active in _active_tasks.items():
        if active.thread is not None and active.thread.is_alive():
            raise RuntimeError(
                f"Task {tid} is already training. "
                f"Only one concurrent task supported (gRPC port collision)."
            )

    stop_event = threading.Event()
    active = ActiveTask(task_id=task_id, stop_event=stop_event)
    _active_tasks[task_id] = active

    thread = threading.Thread(
        target=_run_flwr_server,
        args=(task_dict, participants, db_session_factory, active),
        daemon=True,
    )
    active.thread = thread
    thread.start()
    logger.info("Started flwr server thread for task %d", task_id)


def _run_flwr_server(task_dict: dict, participants: list[dict],
                    db_session_factory, active: ActiveTask) -> None:
    """Run inside background thread: start flwr, wait for completion."""
    server_error = None
    try:
        from flwr.server import ServerConfig, start_server

        from fl_code.fed_core.server_core import build_strategy
        from fl_code.models import TCNConfig, build_tcn

        model = build_tcn(TCNConfig())
        state_keys = list(model.state_dict().keys())
        active.state_keys = state_keys

        expected_clients = [p["client_id"] for p in participants]

        # Update task status to 'training'
        db = db_session_factory()
        try:
            task = db.query(Task).filter(Task.id == active.task_id).first()
            if task:
                task.status = "training"
                task.started_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()

        def on_round_done(row: dict) -> None:
            """Callback from AuditFedAvg after each round: write to DB."""
            db = db_session_factory()
            try:
                audit = AuditRound(
                    task_id=active.task_id,
                    round=row["round"],
                    expected=json.dumps(row["expected"]),
                    joined=json.dumps(row["joined"]),
                    dropped=json.dumps(row["dropped"]),
                    loss=row.get("loss"),
                    client_losses=json.dumps(row.get("client_losses", {})),
                    clip_norm=row.get("clip_norm"),
                    finished_at=datetime.utcnow(),
                )
                db.add(audit)
                db.commit()
            except Exception as e:
                logger.error("Failed to write audit row: %s", e)
            finally:
                db.close()

        flwr_task = {
            "name": task_dict["name"],
            "rounds": task_dict["rounds"],
            "round_timeout": task_dict.get("round_timeout"),
            "checkpoint_dir": None,
            "audit_path": None,
            "expected_clients": expected_clients,
            "deliver_model": True,
            "started_at": str(datetime.utcnow()),
            "cfg": task_dict.get("cfg", {}),
        }

        strategy = build_strategy(flwr_task, state_keys,
                                  on_round_done=on_round_done)

        grpc_port = task_dict.get("grpc_port", 8089)
        try:
            start_server(
                server_address=f"0.0.0.0:{grpc_port}",
                strategy=strategy,
                config=ServerConfig(
                    num_rounds=task_dict["rounds"],
                    round_timeout=task_dict.get("round_timeout"),
                ),
            )
        except Exception as e:
            logger.error("flwr server error for task %d: %s",
                         active.task_id, e)
            server_error = e

        # Extract final model only if no error
        if server_error is None and strategy._last_parameters is not None:
            active.final_tensors = strategy._last_parameters

    except Exception as e:
        logger.error("flwr server setup failed for task %d: %s",
                     active.task_id, e)
        server_error = e

    # Update status based on outcome
    db = db_session_factory()
    try:
        task = db.query(Task).filter(Task.id == active.task_id).first()
        if task:
            task.status = "failed" if server_error else "completed"
            task.finished_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()

    logger.info("flwr server finished for task %d (status: %s)",
                active.task_id, "failed" if server_error else "completed")

    # NOTE: We intentionally keep the _active_tasks entry so that
    # get_final_model() can serve the trained model bytes.
    # In production, implement cleanup_completed_tasks() with TTL.


def cleanup_completed_tasks(max_age_hours: int = 24) -> int:
    """Remove completed tasks older than max_age_hours.

    Returns count of tasks removed.  For the current demo we keep all
    completed tasks so that ``get_final_model()`` continues to work.
    A production deployment would implement TTL-based cleanup here.
    """
    # Demo stub — no cleanup performed
    return 0
