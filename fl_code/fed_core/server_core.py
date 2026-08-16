"""FedAvg strategy + audit log + checkpoints (real implementation in Task 8).

Frozen contract — the App line depends only on these signatures.
"""
from __future__ import annotations


def build_strategy(task: dict, state_keys: list[str],
                   on_round_done=None):
    """task: {"rounds", "round_timeout", "checkpoint_dir", "audit_path",
              "expected_clients", "deliver_model": bool, "cfg": {...}}
    cfg（随每轮 FitIns 下发的扁平训练配置）: {"lr", "batch_size",
              "local_epochs", "dp_mode", "dp_clip", "dp_delta",
              "dp_sigma"|None, "dp_target_epsilon"|None}"""
    raise NotImplementedError("Task 8")


def build_server_app(task: dict, state_keys: list[str]):
    raise NotImplementedError("Task 8")


def make_client_fn(caches: dict, client_ids: list[str],
                   state_keys: list[str], cfg: dict):
    raise NotImplementedError("Task 8")
