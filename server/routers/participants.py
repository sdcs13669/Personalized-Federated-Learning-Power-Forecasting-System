"""Join task with key validation + list participants."""
from __future__ import annotations

import hmac
import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.database import get_db
from server.models import Task, Participant, User
from server.routers.auth import get_current_user_from_header

router = APIRouter(prefix="/api/tasks", tags=["participants"])

FL_SERVER_HOST = os.environ.get("FL_SERVER_HOST", "127.0.0.1")
FL_GRPC_PORT = int(os.environ.get("FL_GRPC_PORT", "8089"))


class JoinRequest(BaseModel):
    key_hash: str
    client_id: str


@router.post("/{task_id}/join")
def join_task(
    task_id: int,
    req: JoinRequest,
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
        "grpc_addr": f"{FL_SERVER_HOST}:{FL_GRPC_PORT}",
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
