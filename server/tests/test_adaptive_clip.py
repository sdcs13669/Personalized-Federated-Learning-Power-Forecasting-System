"""Adaptive clipping: task fields, validation, audit clip_norm."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from server.main import app
from server import database as db_mod


@pytest.fixture(autouse=True)
def setup_db(tmp_path):
    """与 test_results.py 相同的隔离模式：tmp SQLite + dependency_overrides。"""
    db_path = str(tmp_path / "test.db")
    engine = db_mod.get_engine(f"sqlite:///{db_path}")
    db_mod.Base.metadata.create_all(bind=engine)
    db_mod._migrate(engine)
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
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_create_task_with_adaptive_fields():
    token = _register("alice")
    r = client.post("/api/tasks", headers=_auth(token),
                    json={"name": "t", "rounds": 3, "dp_epsilon": 7.5,
                          "dp_clip": 2.0, "dp_adaptive_clip": True,
                          "dp_clip_lr": 0.3,
                          "dp_clip_target_quantile": 0.4,
                          "dp_clip_count_noise": 1.0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dp_adaptive_clip"] is True
    assert body["dp_clip_lr"] == 0.3
    assert body["dp_clip_target_quantile"] == 0.4
    assert body["dp_clip_count_noise"] == 1.0


def test_create_adaptive_without_epsilon_400():
    token = _register("bob")
    r = client.post("/api/tasks", headers=_auth(token),
                    json={"name": "t", "rounds": 3,
                          "dp_adaptive_clip": True})
    assert r.status_code == 400, r.text
    assert "dp_epsilon" in r.json()["detail"]


def test_create_adaptive_with_invalid_hyperparams_400():
    token = _register("carol")
    for body in ({"dp_clip_lr": 0.0},
                 {"dp_clip_target_quantile": 1.5},
                 {"dp_clip_count_noise": -1.0}):
        r = client.post("/api/tasks", headers=_auth(token),
                        json={"name": "t", "rounds": 3, "dp_epsilon": 7.5,
                              "dp_adaptive_clip": True, **body})
        assert r.status_code == 400, r.text


def test_migration_backfills_new_columns(tmp_path):
    """旧库（无新列）启动时被 _migrate 补齐。"""
    from sqlalchemy import text
    engine = db_mod.get_engine(f"sqlite:///{tmp_path}/old.db")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE tasks (id INTEGER PRIMARY KEY, "
                          "name TEXT, creator_id INTEGER, key_hash TEXT, "
                          "rounds INTEGER)"))
        conn.execute(text("CREATE TABLE audit_rounds (id INTEGER PRIMARY "
                          "KEY, task_id INTEGER, round INTEGER)"))
    db_mod._migrate(engine)
    cols = {r[1] for r in engine.connect().exec_driver_sql(
        "PRAGMA table_info(tasks)").fetchall()}
    assert {"dp_adaptive_clip", "dp_clip_lr", "dp_clip_target_quantile",
            "dp_clip_count_noise"} <= cols
    acols = {r[1] for r in engine.connect().exec_driver_sql(
        "PRAGMA table_info(audit_rounds)").fetchall()}
    assert "clip_norm" in acols
    engine.dispose()
