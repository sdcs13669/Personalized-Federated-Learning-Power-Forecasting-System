"""Task 8: 管理端大屏依赖的 server 数据（audit client_epsilons、admin 放行）。"""
import json

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


def _insert_audit(tmp_path, task_id, eps):
    engine = db_mod.get_engine(f"sqlite:///{tmp_path}/test.db")
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db = factory()
    try:
        from server.models import AuditRound
        db.add(AuditRound(
            task_id=task_id, round=1,
            expected=json.dumps(["steel_ind_0"]),
            joined=json.dumps(["steel_ind_0"]),
            dropped=json.dumps([]),
            loss=0.5,
            client_losses=json.dumps({"steel_ind_0": 0.5}),
            client_epsilons=json.dumps({"steel_ind_0": eps}),
        ))
        db.commit()
    finally:
        db.close()
        engine.dispose()


def test_audit_returns_client_epsilons(tmp_path):
    token = _register("alice")
    task = _create_task(token)
    _insert_audit(tmp_path, task["id"], 0.33)
    r = client.get(f"/api/tasks/{task['id']}/audit", headers=_auth(token))
    assert r.status_code == 200
    row = r.json()[0]
    assert row["client_epsilons"] == {"steel_ind_0": 0.33}
    assert row["client_losses"] == {"steel_ind_0": 0.5}
    assert row["expected"] == ["steel_ind_0"]
    assert row["joined"] == ["steel_ind_0"]


def test_admin_can_read_others_audit(tmp_path):
    token = _register("alice")
    task = _create_task(token, name="private")
    _insert_audit(tmp_path, task["id"], 0.33)
    admin = _login_admin()
    r = client.get(f"/api/tasks/{task['id']}/audit", headers=_auth(admin))
    assert r.status_code == 200
    assert r.json()[0]["client_epsilons"] == {"steel_ind_0": 0.33}
    # 普通第三方用户仍被拒
    eve = _register("eve")
    r = client.get(f"/api/tasks/{task['id']}/audit", headers=_auth(eve))
    assert r.status_code == 403


def test_admin_can_read_others_rc_results():
    token = _register("alice")
    task = _create_task(token, name="private2")
    admin = _login_admin()
    r = client.get(f"/api/tasks/{task['id']}/rc-results", headers=_auth(admin))
    assert r.status_code == 200
    assert r.json() == []
    eve = _register("eve")
    r = client.get(f"/api/tasks/{task['id']}/rc-results", headers=_auth(eve))
    assert r.status_code == 403
