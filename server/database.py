"""SQLite database connection and session management."""
from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DB_PATH = os.environ.get("DB_PATH", "data/fl_server.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"


class Base(DeclarativeBase):
    pass


def get_engine(url: str | None = None):
    return create_engine(
        url or DATABASE_URL,
        connect_args={"check_same_thread": False},  # SQLite specific
    )


engine = get_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db() -> Generator[Session]:
    """FastAPI dependency: yields a DB session, auto-closes on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _migrate(target_engine=None) -> None:
    """Idempotent SQLite column backfill (adaptive-clip + role)."""
    from sqlalchemy import text
    eng = target_engine or engine
    with eng.begin() as conn:
        task_cols = {r[1] for r in
                     conn.execute(text("PRAGMA table_info(tasks)")).fetchall()}
        audit_cols = {r[1] for r in
                      conn.execute(text("PRAGMA table_info(audit_rounds)")).fetchall()}
        user_cols = {r[1] for r in
                     conn.execute(text("PRAGMA table_info(users)")).fetchall()}
        if "dp_adaptive_clip" not in task_cols:
            conn.execute(text("ALTER TABLE tasks ADD COLUMN "
                              "dp_adaptive_clip BOOLEAN DEFAULT 0"))
        if "dp_clip_lr" not in task_cols:
            conn.execute(text("ALTER TABLE tasks ADD COLUMN "
                              "dp_clip_lr FLOAT DEFAULT 0.2"))
        if "dp_clip_target_quantile" not in task_cols:
            conn.execute(text("ALTER TABLE tasks ADD COLUMN "
                              "dp_clip_target_quantile FLOAT DEFAULT 0.5"))
        if "dp_clip_count_noise" not in task_cols:
            conn.execute(text("ALTER TABLE tasks ADD COLUMN "
                              "dp_clip_count_noise FLOAT DEFAULT 0.5"))
        if "clip_norm" not in audit_cols:
            conn.execute(text("ALTER TABLE audit_rounds ADD COLUMN "
                              "clip_norm FLOAT"))
        if "role" not in user_cols:
            conn.execute(text("ALTER TABLE users ADD COLUMN "
                              "role TEXT DEFAULT 'user'"))


def seed_admin() -> None:
    """Ensure a default admin account exists (Task 5)."""
    from server.routers.auth import _hash_password
    from server.models import User
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == "admin").first()
        if admin is None:
            db.add(User(username="admin",
                        password_hash=_hash_password("admin123"),
                        role="admin"))
            db.commit()
    finally:
        db.close()


def init_db() -> None:
    """Create all tables (called on server startup)."""
    Base.metadata.create_all(bind=engine)
    _migrate()
    seed_admin()
