"""SQLAlchemy ORM models matching the spec schema."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column, Integer, Text, Float, Boolean, TIMESTAMP, ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from server.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(Text, unique=True, nullable=False)
    password_hash = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    created_tasks = relationship("Task", back_populates="creator")
    participations = relationship("Participant", back_populates="user")


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key_hash = Column(Text, nullable=False)
    rounds = Column(Integer, nullable=False)
    round_timeout = Column(Integer, nullable=True)
    start_at = Column(TIMESTAMP, nullable=True)
    status = Column(Text, default="recruiting")
    dp_epsilon = Column(Float, nullable=True)
    dp_delta = Column(Float, nullable=True)
    dp_clip = Column(Float, nullable=True)
    local_epochs = Column(Integer, default=1)
    batch_size = Column(Integer, default=64)
    dp_adaptive_clip = Column(Boolean, default=False)
    dp_clip_lr = Column(Float, default=0.2)
    dp_clip_target_quantile = Column(Float, default=0.5)
    dp_clip_count_noise = Column(Float, default=0.5)
    grpc_port = Column(Integer, default=8089)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    started_at = Column(TIMESTAMP, nullable=True)
    finished_at = Column(TIMESTAMP, nullable=True)

    creator = relationship("User", back_populates="created_tasks")
    participants = relationship("Participant", back_populates="task")
    audit_rounds = relationship("AuditRound", back_populates="task")


class Participant(Base):
    __tablename__ = "participants"
    __table_args__ = (
        UniqueConstraint("task_id", "client_id", name="uq_task_client"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    client_id = Column(Text, nullable=False)
    registered_at = Column(TIMESTAMP, default=datetime.utcnow)
    status = Column(Text, default="registered")

    task = relationship("Task", back_populates="participants")
    user = relationship("User", back_populates="participations")


class AuditRound(Base):
    __tablename__ = "audit_rounds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    round = Column(Integer, nullable=False)
    expected = Column(Text, nullable=False)
    joined = Column(Text, nullable=False)
    dropped = Column(Text, nullable=False)
    loss = Column(Float, nullable=True)
    client_losses = Column(Text, nullable=True)
    client_epsilons = Column(Text, nullable=True)
    clip_norm = Column(Float, nullable=True)
    finished_at = Column(TIMESTAMP, nullable=True)

    task = relationship("Task", back_populates="audit_rounds")
