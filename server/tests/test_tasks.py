"""Test task CRUD API."""
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


def test_create_task():
    token = _register()
    resp = client.post("/api/tasks", json={
        "name": "demo_task", "rounds": 10,
    }, headers=_auth(token))
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "demo_task"
    assert data["rounds"] == 10
    assert data["status"] == "recruiting"
    assert "key" in data  # plaintext key returned once
    assert len(data["key"]) == 32  # token_hex(16) = 32 chars


def test_list_tasks_square():
    token = _register()
    client.post("/api/tasks", json={
        "name": "task_a", "rounds": 5}, headers=_auth(token))
    client.post("/api/tasks", json={
        "name": "task_b", "rounds": 3}, headers=_auth(token))

    resp = client.get("/api/tasks", headers=_auth(token))
    assert resp.status_code == 200
    tasks = resp.json()
    assert len(tasks) == 2
    names = {t["name"] for t in tasks}
    assert names == {"task_a", "task_b"}


def test_get_task_detail():
    token = _register()
    create_resp = client.post("/api/tasks", json={
        "name": "detail_task", "rounds": 20}, headers=_auth(token))
    task_id = create_resp.json()["id"]

    resp = client.get(f"/api/tasks/{task_id}", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json()["name"] == "detail_task"


def test_cancel_task():
    token = _register()
    create_resp = client.post("/api/tasks", json={
        "name": "cancel_me", "rounds": 5}, headers=_auth(token))
    task_id = create_resp.json()["id"]

    resp = client.post(f"/api/tasks/{task_id}/cancel", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"

    # No longer visible in square
    resp2 = client.get("/api/tasks", headers=_auth(token))
    names = [t["name"] for t in resp2.json()]
    assert "cancel_me" not in names


def test_cancel_not_creator():
    token_a = _register("alice")
    token_b = _register("bob")
    create_resp = client.post("/api/tasks", json={
        "name": "alice_task", "rounds": 5}, headers=_auth(token_a))
    task_id = create_resp.json()["id"]

    resp = client.post(f"/api/tasks/{task_id}/cancel", headers=_auth(token_b))
    assert resp.status_code == 403
