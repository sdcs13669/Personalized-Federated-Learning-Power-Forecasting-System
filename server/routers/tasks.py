"""Task CRUD: create, list (square), detail, cancel."""
from __future__ import annotations

import secrets
import hashlib

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.database import get_db
from server.models import Participant, Task, User
from server.routers.auth import get_current_user_from_header

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


class CreateTaskRequest(BaseModel):
    name: str
    rounds: int
    round_timeout: int | None = None
    start_at: str | None = None
    dp_epsilon: float | None = None
    dp_delta: float | None = None
    dp_clip: float | None = None
    dp_adaptive_clip: bool = False
    dp_clip_lr: float = 0.2
    dp_clip_target_quantile: float = 0.5
    dp_clip_count_noise: float = 0.5
    local_epochs: int = 1
    batch_size: int = 64


def _task_to_dict(task: Task, key: str | None = None) -> dict:
    d = {
        "id": task.id,
        "name": task.name,
        "creator": task.creator.username if task.creator else None,
        "rounds": task.rounds,
        "round_timeout": task.round_timeout,
        "status": task.status,
        "dp_epsilon": task.dp_epsilon,
        "dp_delta": task.dp_delta,
        "dp_clip": task.dp_clip,
        "dp_adaptive_clip": task.dp_adaptive_clip,
        "dp_clip_lr": task.dp_clip_lr,
        "dp_clip_target_quantile": task.dp_clip_target_quantile,
        "dp_clip_count_noise": task.dp_clip_count_noise,
        "local_epochs": task.local_epochs,
        "batch_size": task.batch_size,
        "grpc_port": task.grpc_port,
        "created_at": str(task.created_at) if task.created_at else None,
        "start_at": str(task.start_at) if task.start_at else None,
        "participant_count": len(task.participants) if task.participants else 0,
        "current_round": task.audit_rounds[-1].round if task.audit_rounds else 0,
    }
    if key is not None:
        d["key"] = key
    return d


@router.post("")
def create_task(
    req: CreateTaskRequest,
    user: User = Depends(get_current_user_from_header),
    db: Session = Depends(get_db),
):
    if req.dp_adaptive_clip:
        if req.dp_epsilon is None:
            raise HTTPException(
                status_code=400,
                detail="dp_adaptive_clip requires dp_epsilon (per-client DP mode)")
        if req.dp_clip_lr is None or req.dp_clip_lr <= 0:
            raise HTTPException(status_code=400,
                                detail="dp_clip_lr must be > 0")
        if not (0 < req.dp_clip_target_quantile < 1):
            raise HTTPException(status_code=400,
                                detail="dp_clip_target_quantile must be in (0, 1)")
        if req.dp_clip_count_noise is None or req.dp_clip_count_noise <= 0:
            raise HTTPException(status_code=400,
                                detail="dp_clip_count_noise must be > 0")

    key = secrets.token_hex(16)
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    task = Task(
        name=req.name,
        creator_id=user.id,
        key_hash=key_hash,
        rounds=req.rounds,
        round_timeout=req.round_timeout,
        start_at=req.start_at,
        dp_epsilon=req.dp_epsilon,
        dp_delta=req.dp_delta,
        dp_clip=req.dp_clip,
        dp_adaptive_clip=req.dp_adaptive_clip,
        dp_clip_lr=req.dp_clip_lr,
        dp_clip_target_quantile=req.dp_clip_target_quantile,
        dp_clip_count_noise=req.dp_clip_count_noise,
        local_epochs=req.local_epochs,
        batch_size=req.batch_size,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return _task_to_dict(task, key=key)


@router.get("")
def list_tasks(
    user: User = Depends(get_current_user_from_header),
    db: Session = Depends(get_db),
):
    """Square: list all recruiting tasks."""
    tasks = db.query(Task).filter(Task.status == "recruiting").all()
    return [_task_to_dict(t) for t in tasks]


my_router = APIRouter(prefix="/api/my", tags=["tasks"])


@my_router.get("/tasks")
def my_tasks(
    user: User = Depends(get_current_user_from_header),
    db: Session = Depends(get_db),
):
    """Tasks the current user created or joined (Task 5)."""
    created = db.query(Task).filter(Task.creator_id == user.id).all()
    joined = [p.task for p in db.query(Participant)
              .filter(Participant.user_id == user.id).all() if p.task]
    seen, merged = set(), []
    for t in created + joined:
        if t.id in seen:
            continue
        seen.add(t.id)
        d = _task_to_dict(t)
        d["my_role"] = "creator" if t.creator_id == user.id else "participant"
        merged.append(d)
    merged.sort(key=lambda x: x["id"], reverse=True)
    return merged


@router.get("/{task_id}")
def get_task(
    task_id: int,
    user: User = Depends(get_current_user_from_header),
    db: Session = Depends(get_db),
):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return _task_to_dict(task)


@router.post("/{task_id}/cancel")
def cancel_task(
    task_id: int,
    user: User = Depends(get_current_user_from_header),
    db: Session = Depends(get_db),
):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.creator_id != user.id:
        raise HTTPException(status_code=403, detail="Only the creator can cancel")
    if task.status != "recruiting":
        raise HTTPException(status_code=400, detail="Can only cancel recruiting tasks")
    task.status = "cancelled"
    db.commit()
    db.refresh(task)
    return _task_to_dict(task)
