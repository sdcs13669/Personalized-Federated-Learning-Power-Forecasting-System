"""app/agent.py 的登录凭证缓存：必须始终跟随前端最新 Bearer。

回归的 bug（2026-09-12 实测）：
转发层原写法是「只在缓存为空时才写入 token」，于是一个 agent 生命周期内
只写一次。前端一旦重新登录（token 变了），缓存仍是旧 token，而本地
stage2/stage3 下载全局模型用的正是这份缓存 → HTTP 401：
    "下载全局模型被拒绝（HTTP 401）：请确认本客户端已登录平台……"
页面本身是正常的（前端 token 有效），所以现象极具迷惑性；
而且刷新页面、重新登录都救不回来，只能重启 agent。
"""
import json
import os

import pytest
from fastapi.testclient import TestClient

import app.agent as agent


class _FakeResp:
    """_forward 的最小替身。"""

    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}
        self.content = json.dumps(payload if payload is not None else {}).encode()

    def json(self):
        return json.loads(self.content)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """建一个用临时配置的 agent app，并把 _forward 换成记录器。"""
    cfg = tmp_path / "agent_config.json"
    cfg.write_text(json.dumps({
        "server_url": "http://server.invalid",
        "username": "", "password": "",
        "client_id": "t_client", "local_port": 9999, "data_dir": "",
    }), encoding="utf-8")
    monkeypatch.setattr(agent, "CONFIG_PATH", cfg)

    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<html></html>", encoding="utf-8")

    seen = []

    def fake_forward(method, url, headers, body):
        seen.append({"method": method, "url": url, "headers": dict(headers)})
        return _FakeResp(200, {"ok": True})

    monkeypatch.setattr(agent, "_forward", fake_forward)

    # create_app 会直接写这几个环境变量，测完恢复，避免污染其它测试
    keep = {k: os.environ.get(k)
            for k in ("FL_DATA_DIR", "FL_STAGE_DIR", "FL_RC_WORK")}
    app = agent.create_app(web_dir=str(web), server_url="http://server.invalid")
    client = TestClient(app)
    try:
        yield client, seen
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _cached_token(client):
    return client.get("/local/token").json()["token"]


def test_first_bearer_is_cached(env):
    client, _ = env
    client.post("/api/tasks", headers={"authorization": "Bearer T1"})
    assert _cached_token(client) == "T1"


def test_cache_follows_relogin(env):
    """核心回归：前端重新登录后，缓存必须换成新 token。"""
    client, _ = env
    client.post("/api/tasks", headers={"authorization": "Bearer T1"})
    assert _cached_token(client) == "T1"

    # 用户在页面重新登录，前端拿到新 token T2
    client.post("/api/tasks", headers={"authorization": "Bearer T2"})
    assert _cached_token(client) == "T2", "重新登录后缓存未更新 → 二阶段会 401"


def test_cache_follows_relogin_many_times(env):
    client, _ = env
    for i in range(1, 5):
        client.post("/api/tasks", headers={"authorization": f"Bearer T{i}"})
        assert _cached_token(client) == f"T{i}"


def test_forwarded_header_uses_frontend_token_not_cache(env):
    """转发时透传前端 token；缓存只是给本地 stage2/3 用。"""
    client, seen = env
    client.post("/api/tasks", headers={"authorization": "Bearer T1"})
    client.post("/api/tasks", headers={"authorization": "Bearer T2"})

    assert seen[-1]["headers"]["Authorization"] == "Bearer T2"


def test_no_auth_header_falls_back_to_cached(env):
    """前端没带 Authorization 时，用缓存里的 token 转发。"""
    client, seen = env
    client.post("/api/tasks", headers={"authorization": "Bearer T9"})
    client.post("/api/tasks")           # 不带凭证

    assert seen[-1]["headers"]["Authorization"] == "Bearer T9"


def test_bearer_prefix_is_optional(env):
    """前端若发的是裸 token（无 Bearer 前缀），缓存也要正确。"""
    client, _ = env
    client.post("/api/tasks", headers={"authorization": "T3"})
    assert _cached_token(client) == "T3"


def test_token_absent_when_no_auth_ever(env):
    client, _ = env
    assert _cached_token(client) is None
