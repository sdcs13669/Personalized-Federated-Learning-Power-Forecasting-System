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


# ---------------------------------------------------------------------------
# Task 7: fl_runner on_round_done 审计写 clip_norm
# ---------------------------------------------------------------------------


class _Proxy:
    """最小 ClientProxy 替身：只需 .cid 供 AuditFedAvg 归因。"""

    def __init__(self, cid):
        self.cid = cid


def test_on_round_done_writes_clip_norm(tmp_path):
    """AuditRound 模型可携带 clip_norm（隔离 tmp DB，不污染正式 data/fl_server.db）。"""
    from server.models import AuditRound
    engine = db_mod.get_engine(f"sqlite:///{tmp_path}/model.db")
    db_mod.Base.metadata.create_all(bind=engine)
    db_mod._migrate(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db = factory()
    try:
        row = AuditRound(task_id=1, round=1, expected="[]", joined="[]",
                         dropped="[]", clip_norm=2.5)
        db.add(row)
        db.commit()
        assert row.clip_norm == 2.5
    finally:
        db.close()
        engine.dispose()


def test_run_flwr_server_persists_clip_norm(tmp_path, monkeypatch):
    """端到端（无真实 gRPC）：on_round_done 把 row["clip_norm"] 写入 audit_rounds。

    用假 start_server 驱动一次 aggregate_fit，完整走 _run_flwr_server 的
    真实 on_round_done 闭包 → DB。
    """
    import flwr.server
    from flwr.common import Code, FitRes, Status, ndarrays_to_parameters

    from fl_code.models import TCNConfig, build_tcn
    from server.fl_runner import _run_flwr_server, ActiveTask
    from server.models import Task, User, AuditRound

    engine = db_mod.get_engine(f"sqlite:///{tmp_path}/fl_runner.db")
    db_mod.Base.metadata.create_all(bind=engine)
    db_mod._migrate(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    try:
        db = factory()
        user = User(username="runner_u", password_hash="h")
        db.add(user)
        db.commit()
        task = Task(name="t", creator_id=user.id, rounds=3, key_hash="k")
        db.add(task)
        db.commit()
        task_id = task.id
        db.close()

        def fake_start_server(server_address, strategy, config):
            model = build_tcn(TCNConfig())
            tensors = [v.detach().numpy()
                       for v in model.state_dict().values()]
            results = []
            for cid in ("a", "b"):
                results.append((_Proxy(cid), FitRes(
                    status=Status(code=Code.OK, message=""),
                    parameters=ndarrays_to_parameters(tensors),
                    num_examples=10,
                    metrics={"cid": cid, "loss": 1.0,
                             "dpfedavg_clip_fraction": 0.9})))
            strategy.aggregate_fit(1, results, [])

        monkeypatch.setattr(flwr.server, "start_server", fake_start_server)

        task_dict = {"id": task_id, "name": "t", "rounds": 3,
                     "round_timeout": None,
                     "cfg": {"dp_adaptive_clip": True, "dp_clip": 2.5,
                             "dp_clip_lr": 0.2,
                             "dp_clip_target_quantile": 0.5,
                             "dp_clip_count_noise": 0.5,
                             "dp_mode": "per_client"}}
        participants = [{"client_id": "a"}, {"client_id": "b"}]
        active = ActiveTask(task_id=task_id)
        _run_flwr_server(task_dict, participants, factory, active)

        db = factory()
        try:
            audit = (db.query(AuditRound)
                     .filter(AuditRound.task_id == task_id).one())
            # 初始 dp_clip = 2.5，且是 pre-update 的绑定
            assert audit.clip_norm == 2.5
            assert db.query(Task).get(task_id).status == "completed"
        finally:
            db.close()
    finally:
        engine.dispose()
