"""强制停止训练：fl_runner.stop_training 的杀进程逻辑 + /api/tasks/{id}/stop 端点。

该功能于 2026-09-07 加入（86605df），此前无任何测试覆盖。
这里补齐三种 kill 分支（no_active / killed / already_exited）
以及端点的权限校验与"worker 被杀后状态兜底复位"。
"""
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

import server.fl_runner as fl_runner
from server import database as db_mod
from server.main import app


@pytest.fixture(autouse=True)
def setup_db(tmp_path):
    engine = db_mod.get_engine(f"sqlite:///{tmp_path}/test.db")
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
    yield factory
    app.dependency_overrides.clear()
    fl_runner._active_tasks.clear()
    engine.dispose()


client = TestClient(app)


def _spawn_sleeper():
    """起一个真实的长睡眠子进程，用来扮演 flwr worker。"""
    return subprocess.Popen([sys.executable, "-c",
                             "import time; time.sleep(60)"])


def _register(name):
    r = client.post("/api/auth/register",
                    json={"username": name, "password": "pw"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _create_task(token, name="t"):
    r = client.post("/api/tasks", headers=_auth(token),
                    json={"name": name, "rounds": 3})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _set_status(factory, task_id, status):
    from server.models import Task
    db = factory()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = status
        db.commit()
    finally:
        db.close()


def _get_status(factory, task_id):
    from server.models import Task
    db = factory()
    try:
        return db.query(Task).filter(Task.id == task_id).first().status
    finally:
        db.close()


# ---------------------------------------------------------------------------
# fl_runner.stop_training：三个返回分支
# ---------------------------------------------------------------------------

def test_stop_training_no_active():
    """没有登记过的任务 → no_active。"""
    assert fl_runner.stop_training(12345) == "no_active"


def test_stop_training_kills_running_process():
    """正在跑的进程 → 真被杀掉，且内存句柄被清理。"""
    proc = _spawn_sleeper()
    fl_runner._active_tasks[1] = fl_runner.ActiveTask(task_id=1, proc=proc)
    try:
        assert proc.poll() is None, "测试前置：进程应在运行"
        assert fl_runner.stop_training(1) == "killed"
        assert proc.poll() is not None, "进程应已被终止"
        assert 1 not in fl_runner._active_tasks, "句柄应已清理"
    finally:
        if proc.poll() is None:
            proc.kill()


def test_stop_training_already_exited():
    """登记了但进程早已自己退出 → already_exited，仅清理句柄。"""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=60)
    fl_runner._active_tasks[2] = fl_runner.ActiveTask(task_id=2, proc=proc)
    assert fl_runner.stop_training(2) == "already_exited"
    assert 2 not in fl_runner._active_tasks


def test_stop_training_entry_without_proc():
    """登记了但 proc 为 None → no_active。"""
    fl_runner._active_tasks[3] = fl_runner.ActiveTask(task_id=3, proc=None)
    assert fl_runner.stop_training(3) == "no_active"


# ---------------------------------------------------------------------------
# /api/tasks/{id}/stop 端点
# ---------------------------------------------------------------------------

def test_stop_endpoint_kills_and_resets_status(setup_db):
    """端到端：杀进程 + 状态由 training 复位为 stopped。"""
    factory = setup_db
    token = _register("stopper")
    task_id = _create_task(token)
    _set_status(factory, task_id, "training")
    proc = _spawn_sleeper()
    fl_runner._active_tasks[task_id] = fl_runner.ActiveTask(task_id=task_id,
                                                            proc=proc)
    try:
        r = client.post(f"/api/tasks/{task_id}/stop", headers=_auth(token))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["killed"] is True
        assert body["status"] == "stopped"
        assert "已复位" in body["message"]
        assert _get_status(factory, task_id) == "stopped"
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()


def test_stop_endpoint_no_active_process_still_resets(setup_db):
    """没有活动进程时也要把卡住的 training 兜底复位。"""
    factory = setup_db
    token = _register("stopper2")
    task_id = _create_task(token, "t2")
    _set_status(factory, task_id, "training")

    r = client.post(f"/api/tasks/{task_id}/stop", headers=_auth(token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["killed"] is False
    assert "没有正在运行的训练进程" in body["message"]
    assert _get_status(factory, task_id) == "stopped"


def test_stop_endpoint_forbidden_for_other_user(setup_db):
    """非创建者且非 admin → 403。"""
    owner = _register("owner")
    other = _register("other")
    task_id = _create_task(owner, "t3")

    r = client.post(f"/api/tasks/{task_id}/stop", headers=_auth(other))
    assert r.status_code == 403


def test_stop_endpoint_404_for_missing_task(setup_db):
    token = _register("stopper3")
    r = client.post("/api/tasks/999999/stop", headers=_auth(token))
    assert r.status_code == 404
