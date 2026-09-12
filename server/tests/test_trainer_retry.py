"""flwr 客户端「连不上训练通道」的处理：自动重试 + 失败可见。

回归的坑（2026-09-12 由客户端状态实锤）：
steel_ind_0 的 /local/train-status 显示 round=0、loss=null，而同机另两个
客户端都跑到 round=5 —— 说明它的 flwr 客户端**从头到尾一次都没连上**。
原因：flwr 客户端连不上会立刻退出且不重试（实测 Connection refused），
而服务端 worker 要等「开始训练」才监听 8089；客户端先点就整场缺席。
更糟的是这个失败原本完全静默：/local/start 已返回"训练已启动"，
异常在线程里被吞掉，界面上只表现为"该客户端掉线"。
"""
import flwr.client
import pytest

from app import trainer

CFG = {"client_id": "steel_ind_0", "batch_size": 64, "local_epochs": 1,
       "lr": 0.001, "device": "cpu"}


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    trainer._state.update({"thread": None, "running": False, "round": 0,
                           "loss": None, "grpc_addr": None, "error": None})
    monkeypatch.setattr(trainer, "CONNECT_RETRY_INTERVAL", 0.01)
    yield


def _run(addr="100.84.29.9:8089"):
    trainer._run_flwr(addr, {}, ["weight"], dict(CFG))


def test_retries_until_connect(monkeypatch):
    calls = {"n": 0}

    def fake_start_client(*, server_address, client):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("Connection refused")

    monkeypatch.setattr(flwr.client, "start_client", fake_start_client)
    _run()

    assert calls["n"] == 3, "应在第 3 次连接成功"
    assert trainer._state["running"] is False


def test_gives_up_after_deadline_and_records_error(monkeypatch):
    calls = {"n": 0}

    def always_fail(*, server_address, client):
        calls["n"] += 1
        raise ConnectionError("Connection refused")

    monkeypatch.setattr(flwr.client, "start_client", always_fail)
    monkeypatch.setattr(trainer, "CONNECT_RETRY_SECONDS", 0.05)

    with pytest.raises(ConnectionError):
        _run()

    assert calls["n"] > 1, "放弃前应重试过若干次"
    err = trainer._state["error"]
    assert err and "无法连接训练通道" in err
    assert "开始训练" in err, "错误信息要给出可操作的排查方向"


def test_error_is_surfaced_through_train_status(monkeypatch):
    def always_fail(*, server_address, client):
        raise ConnectionError("Connection refused")

    monkeypatch.setattr(flwr.client, "start_client", always_fail)
    monkeypatch.setattr(trainer, "CONNECT_RETRY_SECONDS", 0.05)

    with pytest.raises(ConnectionError):
        _run()

    st = trainer.get_train_status()
    assert st["running"] is False
    assert st["round"] == 0
    assert "无法连接训练通道" in (st["error"] or "")


def test_no_retry_after_already_participating(monkeypatch):
    """训练中途断连 → 不重试（重连会以新身份插进训练中段）。

    注意：_run_flwr 开头会把 round 重置为 0（每次训练是新的一轮），
    所以这里由 fake start_client 自己在"断连前"把 round 抬起来，
    模拟"已经收到过若干轮指令"的真实情形。
    """
    calls = {"n": 0}

    def fail(*, server_address, client):
        calls["n"] += 1
        trainer._state["round"] = 3      # 已经训练过 3 轮
        raise ConnectionError("connection lost")

    monkeypatch.setattr(flwr.client, "start_client", fail)

    with pytest.raises(ConnectionError):
        _run()

    assert calls["n"] == 1, "已参与过就不应重试"
    assert "无法连接训练通道" in (trainer._state["error"] or "")


def test_receiving_a_round_clears_connect_error():
    """收到第一轮指令说明已连上：清掉连接期告警，避免训练成功后仍报错。"""
    trainer._state["error"] = "连接训练通道失败，自动重试中"
    client = trainer._StatusClient.__new__(trainer._StatusClient)

    class _Inner:
        def fit(self, parameters, config):
            return [], 1, {"loss": 0.5}

    client._inner = _Inner()
    client._cid = "steel_ind_0"
    client.fit([], {"server_round": 1})

    assert trainer._state["error"] is None
    assert trainer._state["round"] == 1
