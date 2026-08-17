"""Test join task with key validation."""
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


def _create_task(token, name="task1"):
    resp = client.post("/api/tasks", json={
        "name": name, "rounds": 5}, headers=_auth(token))
    return resp.json()


def test_join_with_correct_key():
    token_a = _register("alice")
    task = _create_task(token_a)
    key = task["key"]

    token_b = _register("bob")
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    resp = client.post(f"/api/tasks/{task['id']}/join", json={
        "key_hash": key_hash, "client_id": "steel_ind_0"
    }, headers=_auth(token_b))
    assert resp.status_code == 200
    assert "grpc_addr" in resp.json()


def test_join_with_wrong_key():
    token_a = _register("alice")
    task = _create_task(token_a)

    token_b = _register("bob")
    wrong_hash = hashlib.sha256(b"wrong_key").hexdigest()
    resp = client.post(f"/api/tasks/{task['id']}/join", json={
        "key_hash": wrong_hash, "client_id": "steel_ind_0"
    }, headers=_auth(token_b))
    assert resp.status_code == 403


def test_join_duplicate_client_id():
    token_a = _register("alice")
    task = _create_task(token_a)
    key = task["key"]
    key_hash = hashlib.sha256(key.encode()).hexdigest()

    token_b = _register("bob")
    client.post(f"/api/tasks/{task['id']}/join", json={
        "key_hash": key_hash, "client_id": "steel_ind_0"
    }, headers=_auth(token_b))

    token_c = _register("carol")
    resp = client.post(f"/api/tasks/{task['id']}/join", json={
        "key_hash": key_hash, "client_id": "steel_ind_0"
    }, headers=_auth(token_c))
    assert resp.status_code == 409


def test_list_participants():
    token_a = _register("alice")
    task = _create_task(token_a)
    key_hash = hashlib.sha256(task["key"].encode()).hexdigest()

    client.post(f"/api/tasks/{task['id']}/join", json={
        "key_hash": key_hash, "client_id": "steel_ind_0"
    }, headers=_auth(token_a))

    resp = client.get(f"/api/tasks/{task['id']}/participants",
                      headers=_auth(token_a))
    assert resp.status_code == 200
    participants = resp.json()
    assert len(participants) == 1
    assert participants[0]["client_id"] == "steel_ind_0"
