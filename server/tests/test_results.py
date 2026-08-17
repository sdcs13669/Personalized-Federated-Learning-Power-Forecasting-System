"""Test results API: start training, audit query, model download."""
import hashlib
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from server.main import app
from server.database import Base, get_engine, SessionLocal


@pytest.fixture(autouse=True)
def setup_db(tmp_path):
    from server import database as db_mod
    db_path = str(tmp_path / "test.db")
    engine = db_mod.get_engine(f"sqlite:///{db_path}")
    db_mod.Base.metadata.create_all(bind=engine)
    # FIX: use sessionmaker directly instead of SessionLocal(bind=engine)
    # which returns a Session, not a factory
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def override_get_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[db_mod.get_db] = override_get_db
    yield
    app.dependency_overrides.clear()
    engine.dispose()


client = TestClient(app)


def _register(name="alice"):
    resp = client.post("/api/auth/register",
                       json={"username": name, "password": "pw"})
    return resp.json()["access_token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_audit_empty():
    token = _register()
    create_resp = client.post("/api/tasks", json={
        "name": "t", "rounds": 1}, headers=_auth(token))
    task_id = create_resp.json()["id"]
    resp = client.get(f"/api/tasks/{task_id}/audit", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json() == []


def test_model_not_ready():
    token = _register()
    create_resp = client.post("/api/tasks", json={
        "name": "t", "rounds": 1}, headers=_auth(token))
    task_id = create_resp.json()["id"]
    resp = client.get(f"/api/tasks/{task_id}/model", headers=_auth(token))
    assert resp.status_code == 404
