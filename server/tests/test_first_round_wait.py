"""第 1 轮等待参与方接入（AuditFedAvg._wait_for_participants）。

背景：flwr 的 fit_round 在 configure_fit 返回空列表时会直接 cancel 并进入
下一轮，因此"开始训练"时若客户端尚未连上 gRPC，服务端会在毫秒级把全部
轮次空跑完、不产生任何审计行。这里验证修复：

  · 第 1 轮且参与方未到齐  → 阻塞等待（有超时上限）
  · 第 1 轮且参与方已到齐  → 不等待
  · 第 2 轮起              → 绝不等待（掉线容错的保证）
  · 非 SimpleClientManager → 不阻塞、不报错（兼容 flwr 仿真路径）
"""
from flwr.common import ndarrays_to_parameters
from flwr.common.typing import Parameters

from fl_code.fed_core.server_core import build_strategy
from fl_code.models import TCNConfig, build_tcn


class FakeClientManager:
    """SimpleClientManager 的最小替身：只记录 wait_for 调用。"""

    def __init__(self, available_cids=(), has_wait=True):
        self.available_cids = list(available_cids)
        self.has_wait = has_wait
        self.wait_calls = []

    def all(self):
        return {cid: object() for cid in self.available_cids}

    def num_available(self):
        return len(self.available_cids)

    def wait_for(self, num_clients, timeout=None):
        self.wait_calls.append((num_clients, timeout))
        return True


class BareClientManager:
    """没有 num_available / wait_for 的替身（模拟仿真路径的 manager）。"""

    def all(self):
        return {}


def _make_strategy(expected, deliver_model=True, **cfg_overrides):
    """注意 deliver_model 默认 True —— 与真实项目一致
    （fl_server_worker 固定传 True，fraction_evaluate=1.0，
    否则 configure_evaluate 会被短路，测不到阻塞问题）。"""
    keys = list(build_tcn(TCNConfig()).state_dict().keys())
    params = ndarrays_to_parameters(
        [v.detach().numpy() for v in build_tcn(TCNConfig()).state_dict().values()])
    cfg = {"dp_adaptive_clip": False, "dp_clip": 1.0, "dp_mode": "none"}
    cfg.update(cfg_overrides)
    task = {
        "name": "t", "rounds": 3, "round_timeout": None,
        "checkpoint_dir": None, "audit_path": None,
        "expected_clients": list(expected), "deliver_model": deliver_model,
        "started_at": "t", "cfg": cfg,
    }
    strategy = build_strategy(task, keys)
    assert isinstance(strategy.initial_parameters, Parameters)
    return strategy, params


def test_first_round_waits_when_participants_missing():
    strategy, params = _make_strategy(["a", "b", "c"])
    mgr = FakeClientManager(available_cids=[])

    strategy.configure_fit(1, params, mgr)

    assert mgr.wait_calls == [(3, 300)], mgr.wait_calls


def test_first_round_timeout_is_configurable():
    strategy, params = _make_strategy(["a", "b"], first_round_timeout=45)
    mgr = FakeClientManager(available_cids=[])

    strategy.configure_fit(1, params, mgr)

    assert mgr.wait_calls == [(2, 45)], mgr.wait_calls


def test_first_round_does_not_wait_when_all_present():
    strategy, params = _make_strategy(["a", "b"])
    mgr = FakeClientManager(available_cids=["a", "b"])

    strategy.configure_fit(1, params, mgr)

    assert mgr.wait_calls == [], mgr.wait_calls


def test_later_rounds_never_wait():
    """掉线容错的关键：第 2 轮起即使参与方不全也不能阻塞。"""
    strategy, params = _make_strategy(["a", "b", "c"])
    mgr = FakeClientManager(available_cids=["a"])   # 只有 1/3 在线（掉线了）

    strategy.configure_fit(2, params, mgr)
    strategy.configure_fit(3, params, mgr)

    assert mgr.wait_calls == [], mgr.wait_calls


def test_wait_tolerates_manager_without_wait_for():
    """仿真路径的 manager 没有 wait_for，不应报错。"""
    strategy, params = _make_strategy(["a", "b"])
    mgr = BareClientManager()

    instructions = strategy.configure_fit(1, params, mgr)

    assert instructions == []


def test_configure_fit_targets_every_online_client():
    """回归：configure_fit 仍然把指令发给「所有在线客户端」（不是采样）。"""
    strategy, params = _make_strategy(["a", "b"])
    mgr = FakeClientManager(available_cids=["a", "b"])

    instructions = strategy.configure_fit(1, params, mgr)

    assert len(instructions) == 2


# ---------------------------------------------------------------------------
# configure_evaluate：原生 FedAvg 会 sample() → wait_for() 阻塞，
# 导致"每轮丢第 1 轮"和"掉线后整体卡死"，因此必须同样重写。
# ---------------------------------------------------------------------------

def test_configure_evaluate_does_not_sample_or_wait():
    strategy, params = _make_strategy(["a", "b", "c"])
    mgr = FakeClientManager(available_cids=["a"])     # 只有 1/3 在线

    instructions = strategy.configure_evaluate(1, params, mgr)

    assert mgr.wait_calls == [], "evaluate 阶段不得调用 wait_for（会卡死训练）"
    assert len(instructions) == 1, "应发给当前在线的那 1 个客户端"


def test_configure_evaluate_targets_every_online_client():
    strategy, params = _make_strategy(["a", "b"])
    mgr = FakeClientManager(available_cids=["a", "b"])

    instructions = strategy.configure_evaluate(2, params, mgr)

    assert len(instructions) == 2
    assert mgr.wait_calls == []


def test_configure_evaluate_skipped_when_fraction_zero():
    """deliver_model=False → fraction_evaluate=0 → 不做联邦评估。"""
    strategy, params = _make_strategy(["a", "b"])
    strategy.fraction_evaluate = 0.0
    mgr = FakeClientManager(available_cids=["a", "b"])

    assert strategy.configure_evaluate(1, params, mgr) == []
