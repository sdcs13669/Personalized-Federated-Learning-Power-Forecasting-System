"""Join task with key validation + list participants."""
from __future__ import annotations

import hmac
import os

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.database import get_db
from server.models import Task, Participant, User
from server.routers.auth import get_current_user_from_header

router = APIRouter(prefix="/api/tasks", tags=["participants"])

FL_GRPC_PORT = int(os.environ.get("FL_GRPC_PORT", "8089"))

# 这些主机名对「客户端」不可达：通配地址、回环、空值
_UNUSABLE_GRPC_HOSTS = {"", "0.0.0.0", "127.0.0.1", "localhost",
                        "::", "[::]", "::1"}


def _grpc_host(request: Request | None = None) -> str:
    """给出**客户端能连上**的 gRPC 主机名（多机演示的关键）。

    坑：`FL_SERVER_HOST` 默认 "127.0.0.1"（裸机），docker-compose 里又是
    "0.0.0.0" —— 这两个对客户端都不可用：加入任务时服务端会把
    "127.0.0.1:8089" 下发给客户端，客户端于是去连**自己机器**的 8089，
    连不上后 flwr 客户端直接退出（实测：连不上不会重试），
    结果该参与方在审计里每一轮都被记为「掉线」。

    策略：显式配置优先（且必须不是通配/回环地址）；否则用客户端**本次请求
    所用的主机名**——客户端既然能通过它访问服务端，它对客户端就一定可达。
    这样双机 / Tailscale 环境零配置即可工作。
    """
    host = (os.environ.get("FL_SERVER_HOST") or "").strip()
    if host not in _UNUSABLE_GRPC_HOSTS:
        return host
    if request is not None:
        req_host = (request.url.hostname or "").strip()
        if req_host not in _UNUSABLE_GRPC_HOSTS:
            return req_host
    return host or "127.0.0.1"


class JoinRequest(BaseModel):
    key_hash: str
    client_id: str


@router.post("/{task_id}/join")
def join_task(
    task_id: int,
    req: JoinRequest,
    request: Request,
    user: User = Depends(get_current_user_from_header),
    db: Session = Depends(get_db),
):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != "recruiting":
        raise HTTPException(status_code=400,
                            detail="Task is not accepting participants")
    if not hmac.compare_digest(req.key_hash, task.key_hash or ""):
        raise HTTPException(status_code=403, detail="Invalid key")

    existing = db.query(Participant).filter(
        Participant.task_id == task_id,
        Participant.client_id == req.client_id,
    ).first()
    if existing:
        raise HTTPException(status_code=409,
                            detail="client_id already registered in this task")

    participant = Participant(
        task_id=task_id,
        user_id=user.id,
        client_id=req.client_id,
    )
    db.add(participant)
    db.commit()
    db.refresh(participant)
    return {
        "participant_id": participant.id,
        "grpc_addr": f"{_grpc_host(request)}:{FL_GRPC_PORT}",
        "client_id": req.client_id,
    }


@router.get("/{task_id}/participants")
def list_participants(
    task_id: int,
    user: User = Depends(get_current_user_from_header),
    db: Session = Depends(get_db),
):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    participants = db.query(Participant).filter(
        Participant.task_id == task_id).all()
    return [
        {
            "id": p.id,
            "client_id": p.client_id,
            "username": p.user.username if p.user else None,
            "status": p.status,
            "registered_at": str(p.registered_at) if p.registered_at else None,
        }
        for p in participants
    ]
