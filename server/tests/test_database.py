"""Verify database creation and ORM models."""
from server.database import Base, get_engine, SessionLocal
from server.models import User, Task, Participant, AuditRound


def test_tables_created():
    engine = get_engine("sqlite:///test_tables.db")
    Base.metadata.create_all(bind=engine)
    from sqlalchemy import inspect
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    assert "users" in table_names
    assert "tasks" in table_names
    assert "participants" in table_names
    assert "audit_rounds" in table_names
    engine.dispose()
    import os
    os.remove("test_tables.db")


def test_create_user():
    engine = get_engine("sqlite:///test_user.db")
    Base.metadata.create_all(bind=engine)
    session = SessionLocal(bind=engine)
    user = User(username="alice", password_hash="hashed_pw")
    session.add(user)
    session.commit()
    fetched = session.query(User).filter_by(username="alice").first()
    assert fetched is not None
    assert fetched.username == "alice"
    session.close()
    engine.dispose()
    import os
    os.remove("test_user.db")
