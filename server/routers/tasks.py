"""Task CRUD: create, list (square), detail, cancel, delete."""
from __future__ import annotations

import secrets
import hashlib
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.database import get_db
from server.models import AuditRound, Participant, RcResult, Task, User
from server.routers.auth import get_current_user_from_header
from server.fl_runner import MODEL_DIR, get_task_status

router = APIRouter(prefix="/api/tasks", tags=["tasks"])
my_router = APIRouter(prefix="/api", tags=["tasks"])


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
    rounds = [r.round for r in task.audit_rounds] if task.audit_rounds else []
    d["current_round"] = max(rounds) if rounds else 0
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
    """Square: recruiting tasks for users; all statuses for admins."""
    if user.role == "admin":
        tasks = db.query(Task).order_by(Task.created_at.desc()).all()
    else:
        tasks = db.query(Task).filter(Task.status == "recruiting").all()
    return [_task_to_dict(t) for t in tasks]


@my_router.get("/my/tasks")
def my_tasks(
    user: User = Depends(get_current_user_from_header),
    db: Session = Depends(get_db),
):
    """Tasks I created or joined, each with my_role."""
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


# ---------------------------------------------------------------------------
# 删除任务（录制/演示前清理历史任务用）
# ---------------------------------------------------------------------------

_RC_UPLOADS = Path(__file__).resolve().parent.parent / "rc_uploads"


def _remove_task_files(task_id: int) -> None:
    """删除任务落盘的产物：最终模型、worker 日志、RC 对比图。"""
    targets = [MODEL_DIR / f"fl_model_{task_id}.pkl",
               MODEL_DIR / f"fl_worker_{task_id}.log"]
    if _RC_UPLOADS.exists():
        targets += list(_RC_UPLOADS.glob(f"task{task_id}_*.png"))
    for p in targets:
        try:
            p.unlink()
        except OSError:
            pass   # 不存在或被占用都不阻塞删库


def _purge_task_data(db: Session, task: Task) -> None:
    """删任务本体 + 关联行 + 落盘产物。

    模型的外键没配 cascade，必须显式删 audit_rounds / participants /
    rc_results，否则会留孤儿行（管理端统计与审计导出会读到脏数据）。
    """
    db.query(AuditRound).filter(AuditRound.task_id == task.id).delete()
    db.query(Participant).filter(Participant.task_id == task.id).delete()
    db.query(RcResult).filter(RcResult.task_id == task.id).delete()
    db.delete(task)
    db.commit()
    _remove_task_files(task.id)


@router.delete("/{task_id}")
def delete_task(
    task_id: int,
    user: User = Depends(get_current_user_from_header),
    db: Session = Depends(get_db),
):
    """删除单个任务（创建者或管理员）。训练中的任务须先强制停止。"""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.creator_id != user.id and user.role != "admin":
        raise HTTPException(status_code=403,
                            detail="Only the creator or admin can delete")
    if get_task_status(task_id) == "training":
        raise HTTPException(
            status_code=400,
            detail="该任务正在训练中，请先点「强制停止训练」再删除")

    _purge_task_data(db, task)
    return {"ok": True, "deleted_id": task_id,
            "message": f"任务 {task_id} 已删除（含审计与参与记录）"}


class PurgeRequest(BaseModel):
    include_recruiting: bool = False


@my_router.post("/tasks/purge")
def purge_tasks(
    req: PurgeRequest,
    user: User = Depends(get_current_user_from_header),
    db: Session = Depends(get_db),
):
    """批量清理历史任务（录制/演示前把界面清干净）。

    默认只清终态任务（已完成/失败/已停止/已取消）；
    include_recruiting=true 连「招募中」的一起清。
    训练中的任务一律跳过——必须先强制停止，避免删掉正在写的审计。
    """
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")

    # 范围里包含 training：区分"真在训练"与"历史卡死的 training 残留"
    # 靠 get_task_status()（看有没有真实子进程）判断——卡死残留应当可清，
    # 否则录制时界面上会永远挂着几个假的「训练中」任务。
    statuses = ["completed", "failed", "stopped", "cancelled", "training"]
    if req.include_recruiting:
        statuses.append("recruiting")

    tasks = db.query(Task).filter(Task.status.in_(statuses)).all()
    deleted, skipped = [], []
    for t in tasks:
        if get_task_status(t.id) == "training":
            skipped.append(t.id)
            continue
        _purge_task_data(db, t)
        deleted.append(t.id)

    msg = f"已清理 {len(deleted)} 个历史任务"
    if skipped:
        msg += f"，跳过 {len(skipped)} 个训练中的任务"
    return {"ok": True, "deleted": len(deleted), "deleted_ids": deleted,
            "skipped_training": skipped, "message": msg}
