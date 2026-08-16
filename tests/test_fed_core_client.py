"""train_client must reproduce train_baseline's existing behavior."""
import numpy as np
import torch

from fl_code.fed_core.client_core import FedClient, train_client
from fl_code.fed_core.data import load_client_cache
from fl_code.models import TCNConfig, build_tcn


def _cfg(rounds=1):
    return {"lr": 0.001, "batch_size": 16, "local_epochs": 1,
            "device": "cpu", "round": 1, "rounds": rounds,
            "budget_path": None}


def test_plain_round_runs_and_returns_shapes():
    cache = load_client_cache("steel_ind_0", stride=6, max_seqs=1)
    model = build_tcn(TCNConfig())
    keys = list(model.state_dict().keys())
    result = train_client(
        [v.detach().numpy() for v in model.state_dict().values()],
        keys, model, cache["train_ds"], cache["n_train"], _cfg(), None)
    assert result["n_train"] == cache["n_train"]
    assert len(result["tensors"]) == len(keys)
    assert np.isfinite(result["loss"])
    assert result["eps"] is None and result["sigma"] is None


def test_dp_round_eps_matches_anchor():
    cache = load_client_cache("steel_ind_0", stride=6, max_seqs=1)
    model = build_tcn(TCNConfig())
    keys = list(model.state_dict().keys())
    dp = {"mode": "uniform", "sigma": 1.0, "clipping_norm": 1.0,
          "delta": 1e-5, "target_epsilon": 0.0}
    result = train_client(
        [v.detach().numpy() for v in model.state_dict().values()],
        keys, model, cache["train_ds"], cache["n_train"], _cfg(), dp)
    assert result["sigma"] == 1.0
    # 与 PLD 会计直接计算一致（回归锚点）
    from fl_code.fed_core.accounting import dp_epsilon
    assert abs(result["eps"] - dp_epsilon(
        cache["n_train"], 16, 1, 1, 1.0, 1e-5)) < 1e-9


def test_per_client_mode_caches_sigma(monkeypatch):
    """per_client σ is derived once per client, then cached across rounds."""
    import fl_code.fed_core.client_core as cc

    calls = []
    real_sigma_for_epsilon = cc.sigma_for_epsilon

    def counting_sigma_for_epsilon(n_train, batch_size, local_epochs,
                                   rounds, delta, target):
        calls.append((n_train, batch_size, local_epochs, rounds,
                      delta, target))
        return real_sigma_for_epsilon(n_train, batch_size, local_epochs,
                                      rounds, delta, target)

    monkeypatch.setattr(cc, "sigma_for_epsilon", counting_sigma_for_epsilon)

    cache = load_client_cache("steel_ind_0", stride=6, max_seqs=1)
    keys = list(build_tcn(TCNConfig()).state_dict().keys())
    client = FedClient(cache, keys, {"lr": 0.001, "batch_size": 16,
                                     "local_epochs": 1, "device": "cpu",
                                     "rounds": 2, "budget_path": None})
    common = {"dp_mode": "per_client", "dp_clip": 1.0, "dp_delta": 1e-5,
              "dp_target_epsilon": 7.5, "rounds": 2}
    tensors = client.get_parameters({})
    # Same long-lived client across rounds (App line pattern): FedClient.fit
    # builds a fresh model per round, so torch.func-marked modules never
    # pass through Module._apply twice — this call sequence is the
    # regression test for that crash.
    m1 = client.fit(tensors, {**common, "server_round": 1})[2]
    m2 = client.fit(tensors, {**common, "server_round": 2})[2]
    # σ derived once, cached across rounds (not recomputed every round)
    assert len(calls) == 1
    assert np.isfinite(m1["eps"]) and np.isfinite(m2["eps"])
    # per-round budget accumulates: ε after 2 rounds > ε after 1 round
    assert m2["eps"] > m1["eps"]
