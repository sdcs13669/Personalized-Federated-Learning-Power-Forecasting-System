"""加入任务时下发给客户端的 gRPC 地址：必须是**客户端能连上**的地址。

回归的坑（2026-09-12 定位）：
`FL_SERVER_HOST` 默认 "127.0.0.1"（裸机 run_server.bat 从不设置它），
docker-compose 里又设成 "0.0.0.0"。两者对客户端都不可用 —— 客户端会去连
自己机器的 8089，连不上后 flwr 客户端直接退出（实测：连不上不重试），
于是该参与方在审计里**每一轮都被记为掉线**。
"""
import hashlib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

import server.routers.participants as participants
from server.main import app


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    from server import database as db_mod
    engine = db_mod.get_engine(f"sqlite:///{tmp_path}/test.db")
    db_mod.Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def override_get_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[db_mod.get_db] = override_get_db
    # 每个用例从「未配置」开始，避免外部环境干扰
    monkeypatch.delenv("FL_SERVER_HOST", raising=False)
    yield
    app.dependency_overrides.clear()
    engine.dispose()


client = TestClient(app)


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _join_with_host(host_header: str):
    """建任务 → 加入，并带上指定的 Host 头，返回 grpc_addr。"""
    tok = client.post("/api/auth/register",
                      json={"username": f"u{abs(hash(host_header)) % 10**6}",
                            "password": "pw"}).json()["access_token"]
    task = client.post("/api/tasks", json={"name": "t", "rounds": 3},
                       headers=_auth(tok)).json()
    key_hash = hashlib.sha256(task["key"].encode()).hexdigest()
    resp = client.post(f"/api/tasks/{task['id']}/join",
                       json={"key_hash": key_hash, "client_id": "c1"},
                       headers={**_auth(tok), "Host": host_header})
    assert resp.status_code == 200, resp.text
    return resp.json()["grpc_addr"], tok


# ---------------------------------------------------------------------------
# _grpc_host 纯函数
# ---------------------------------------------------------------------------

def test_env_set_to_reachable_host_wins(monkeypatch):
    monkeypatch.setenv("FL_SERVER_HOST", "10.0.0.5")
    assert participants._grpc_host(None) == "10.0.0.5"


def test_loopback_env_falls_back_to_request_host(monkeypatch):
    """★最关键：裸机 run_server.bat 从不设 FL_SERVER_HOST，默认就是 127.0.0.1。
    此时必须改用客户端请求所用的主机名，否则客户端会去连自己机器的 8089。"""
    monkeypatch.setenv("FL_SERVER_HOST", "127.0.0.1")
    addr, _ = _join_with_host("100.84.29.9:8000")
    assert addr == "100.84.29.9:8089"


def test_localhost_env_falls_back_to_request_host(monkeypatch):
    monkeypatch.setenv("FL_SERVER_HOST", "localhost")
    addr, _ = _join_with_host("100.84.29.9:8000")
    assert addr == "100.84.29.9:8089"


def test_wildcard_env_falls_back_to_request_host(monkeypatch):
    monkeypatch.setenv("FL_SERVER_HOST", "0.0.0.0")
    addr, _ = _join_with_host("100.84.29.9:8000")
    assert addr == "100.84.29.9:8089"


# ---------------------------------------------------------------------------
# 端点行为
# ---------------------------------------------------------------------------

def test_uses_request_host_when_env_unset():
    """核心：客户端用 100.84.29.9 访问服务端 → 就下发 100.84.29.9:8089。"""
    addr, _ = _join_with_host("100.84.29.9:8000")
    assert addr == "100.84.29.9:8089"


def test_different_client_sees_its_own_reachable_host():
    addr1, _ = _join_with_host("100.84.29.9:8000")
    addr2, _ = _join_with_host("192.168.1.20:8000")
    assert addr1 == "100.84.29.9:8089"
    assert addr2 == "192.168.1.20:8089"


def test_localhost_access_yields_loopback():
    """同机访问（localhost）时下发 127.0.0.1 才是对的。"""
    addr, _ = _join_with_host("localhost:8000")
    assert addr == "127.0.0.1:8089"


def test_explicit_reachable_env_overrides_request_host(monkeypatch):
    monkeypatch.setenv("FL_SERVER_HOST", "10.9.9.9")
    addr, _ = _join_with_host("100.84.29.9:8000")
    assert addr == "10.9.9.9:8089"


def test_grpc_port_is_configurable(monkeypatch):
    monkeypatch.setattr(participants, "FL_GRPC_PORT", 18089)
    addr, _ = _join_with_host("100.84.29.9:8000")
    assert addr == "100.84.29.9:18089"
