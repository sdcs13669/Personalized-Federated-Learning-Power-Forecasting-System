"""Task 5: role 字段、seed admin、我的任务、当前轮次。"""
import hashlib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from server.main import app
from server import database as db_mod


@pytest.fixture(autouse=True)
def setup_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    engine = db_mod.get_engine(f"sqlite:///{db_path}")
    db_mod.Base.metadata.create_all(bind=engine)
    db_mod._migrate(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db_mod.seed_admin(session_factory=factory)

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
    r = client.post("/api/auth/register",
                    json={"username": name, "password": "pw123456"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _login_admin():
    r = client.post("/api/auth/login",
                    json={"username": "admin", "password": "admin123"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _create_task(token, name="t", rounds=5):
    r = client.post("/api/tasks", headers=_auth(token),
                    json={"name": name, "rounds": rounds})
    assert r.status_code == 200, r.text
    return r.json()


def _insert_audit_round(tmp_path, task_id, round_num):
    engine = db_mod.get_engine(f"sqlite:///{tmp_path}/test.db")
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db = factory()
    try:
        from server.models import AuditRound
        db.add(AuditRound(task_id=task_id, round=round_num,
                          expected="[]", joined="[]", dropped="[]", loss=0.5))
        db.commit()
    finally:
        db.close()
        engine.dispose()


def test_me_returns_role():
    token = _register("alice")
    r = client.get("/api/me", headers=_auth(token))
    assert r.status_code == 200
    assert r.json()["role"] == "user"


def test_admin_login_has_admin_role():
    token = _login_admin()
    r = client.get("/api/me", headers=_auth(token))
    assert r.status_code == 200
    assert r.json()["role"] == "admin"


def test_seed_admin_idempotent(tmp_path):
    engine = db_mod.get_engine(f"sqlite:///{tmp_path}/seed.db")
    db_mod.Base.metadata.create_all(bind=engine)
    db_mod._migrate(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db_mod.seed_admin(session_factory=factory)
    db_mod.seed_admin(session_factory=factory)
    db = factory()
    try:
        from server.models import User
        admins = db.query(User).filter(User.username == "admin").all()
        assert len(admins) == 1
        assert admins[0].role == "admin"
    finally:
        db.close()
        engine.dispose()


def test_my_tasks_includes_created_and_joined():
    bob = _register("bob")
    task = _create_task(bob, name="t1")
    r = client.get("/api/my/tasks", headers=_auth(bob))
    assert r.status_code == 200
    assert any(t["id"] == task["id"] and t["my_role"] == "creator"
               for t in r.json())

    carol = _register("carol")
    key_hash = hashlib.sha256(task["key"].encode()).hexdigest()
    r = client.post(f"/api/tasks/{task['id']}/join", headers=_auth(carol),
                    json={"key_hash": key_hash, "client_id": "steel_ind_0"})
    assert r.status_code == 200, r.text
    r = client.get("/api/my/tasks", headers=_auth(carol))
    assert r.status_code == 200
    assert any(t["id"] == task["id"] and t["my_role"] == "participant"
               for t in r.json())


def test_admin_lists_all_statuses():
    dave = _register("dave")
    task = _create_task(dave, name="t2")
    r = client.post(f"/api/tasks/{task['id']}/cancel", headers=_auth(dave))
    assert r.status_code == 200

    r = client.get("/api/tasks", headers=_auth(dave))
    names = [t["name"] for t in r.json()]
    assert "t2" not in names

    admin = _login_admin()
    r = client.get("/api/tasks", headers=_auth(admin))
    names = [t["name"] for t in r.json()]
    assert "t2" in names


def test_current_round_in_detail_and_my_tasks(tmp_path):
    token = _register("alice")
    task = _create_task(token, name="t3")
    r = client.get(f"/api/tasks/{task['id']}", headers=_auth(token))
    assert r.json()["current_round"] == 0

    _insert_audit_round(tmp_path, task["id"], 1)
    _insert_audit_round(tmp_path, task["id"], 2)

    r = client.get(f"/api/tasks/{task['id']}", headers=_auth(token))
    assert r.json()["current_round"] == 2

    r = client.get("/api/my/tasks", headers=_auth(token))
    t = next(x for x in r.json() if x["id"] == task["id"])
    assert t["current_round"] == 2
