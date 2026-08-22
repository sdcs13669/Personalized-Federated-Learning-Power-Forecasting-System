"""Results API: start training, audit query, model download."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm import Session

from server.database import get_db, SessionLocal
from server.models import Task, AuditRound, Participant, RcResult, User
from server.routers.auth import get_current_user_from_header
from server.fl_runner import start_training, get_final_model

router = APIRouter(prefix="/api/tasks", tags=["results"])

RC_UPLOADS_DIR = Path(__file__).resolve().parent.parent / "rc_uploads"


@router.post("/{task_id}/start")
def start_task(
    task_id: int,
    user: User = Depends(get_current_user_from_header),
    db: Session = Depends(get_db),
):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.creator_id != user.id:
        raise HTTPException(status_code=403, detail="Only the creator can start")
    if task.status != "recruiting":
        raise HTTPException(status_code=400, detail="Task is not in recruiting state")

    participants = db.query(Participant).filter(
        Participant.task_id == task_id).all()
    if not participants:
        raise HTTPException(status_code=400,
                            detail="No participants registered yet")

    task_dict = {
        "id": task.id,
        "name": task.name,
        "rounds": task.rounds,
        "round_timeout": task.round_timeout,
        "grpc_port": task.grpc_port,
        "cfg": {
            "lr": 0.001,
            "batch_size": task.batch_size,
            "local_epochs": task.local_epochs,
            "dp_mode": "per_client" if task.dp_epsilon else "none",
            "dp_clip": task.dp_clip or 1.0,
            "dp_delta": task.dp_delta or 1e-5,
            "dp_sigma": None,
            "dp_target_epsilon": task.dp_epsilon,
            "dp_adaptive_clip": task.dp_adaptive_clip or False,
            "dp_clip_lr": task.dp_clip_lr or 0.2,
            "dp_clip_target_quantile": task.dp_clip_target_quantile or 0.5,
            "dp_clip_count_noise": task.dp_clip_count_noise or 0.5,
        },
    }
    participant_list = [
        {"client_id": p.client_id, "user_id": p.user_id}
        for p in participants
    ]

    start_training(task_dict, participant_list, SessionLocal)
    return {"status": "training", "message": "Training started"}


@router.get("/{task_id}/audit")
def get_audit(
    task_id: int,
    user: User = Depends(get_current_user_from_header),
    db: Session = Depends(get_db),
):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Check user is a participant or the creator
    is_participant = db.query(Participant).filter(
        Participant.task_id == task_id,
        Participant.user_id == user.id,
    ).first()
    if not is_participant and task.creator_id != user.id:
        raise HTTPException(status_code=403, detail="Not a participant")

    rounds = db.query(AuditRound).filter(
        AuditRound.task_id == task_id
    ).order_by(AuditRound.round).all()
    return [
        {
            "round": r.round,
            "expected": json.loads(r.expected),
            "joined": json.loads(r.joined),
            "dropped": json.loads(r.dropped),
            "loss": r.loss,
            "client_losses": json.loads(r.client_losses) if r.client_losses else {},
            "clip_norm": r.clip_norm,
            "finished_at": str(r.finished_at) if r.finished_at else None,
        }
        for r in rounds
    ]


@router.get("/{task_id}/model")
def download_model(
    task_id: int,
    user: User = Depends(get_current_user_from_header),
    db: Session = Depends(get_db),
):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Check user is a participant or the creator
    is_participant = db.query(Participant).filter(
        Participant.task_id == task_id,
        Participant.user_id == user.id,
    ).first()
    if not is_participant and task.creator_id != user.id:
        raise HTTPException(status_code=403, detail="Not a participant")

    model_bytes = get_final_model(task_id)
    if model_bytes is None:
        raise HTTPException(status_code=404,
                            detail="Model not ready (training not completed)")
    return Response(
        content=model_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition":
                 f"attachment; filename=model_task{task_id}.pkl"},
    )


@router.post("/{task_id}/rc-result")
def upload_rc_result(
    task_id: int,
    client_id: str = Form(...),
    wape_global: float = Form(...),
    wape_rc: float = Form(...),
    png: UploadFile | None = File(None),
    user: User = Depends(get_current_user_from_header),
    db: Session = Depends(get_db),
):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    is_participant = db.query(Participant).filter(
        Participant.task_id == task_id,
        Participant.client_id == client_id,
    ).first()
    if not is_participant:
        raise HTTPException(status_code=403, detail="Not a participant")

    # 覆盖式：同 client 重复上传则更新
    rc = db.query(RcResult).filter(
        RcResult.task_id == task_id,
        RcResult.client_id == client_id,
    ).first()
    if rc is None:
        rc = RcResult(task_id=task_id, client_id=client_id)
        db.add(rc)
    rc.wape_global = wape_global
    rc.wape_rc = wape_rc
    if png is not None:
        RC_UPLOADS_DIR.mkdir(exist_ok=True)
        out = RC_UPLOADS_DIR / f"task{task_id}_{client_id}.png"
        with out.open("wb") as f:
            shutil.copyfileobj(png.file, f)
        rc.png_path = f"/rc_uploads/task{task_id}_{client_id}.png"
    db.commit()
    return {"ok": True}


@router.get("/{task_id}/rc-results")
def list_rc_results(
    task_id: int,
    user: User = Depends(get_current_user_from_header),
    db: Session = Depends(get_db),
):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    is_participant = db.query(Participant).filter(
        Participant.task_id == task_id,
        Participant.user_id == user.id,
    ).first()
    if not is_participant and task.creator_id != user.id:
        raise HTTPException(status_code=403, detail="Not a participant")
    rows = db.query(RcResult).filter(RcResult.task_id == task_id).all()
    return [
        {"client_id": r.client_id,
         "wape_global": r.wape_global,
         "wape_rc": r.wape_rc,
         "png_url": r.png_path,
         "created_at": str(r.created_at) if r.created_at else None}
        for r in rows
    ]
