import pytest

from fl_code.fed_core.accounting import adaptive_sigma_train


def test_zero_count_noise_rejected():
    with pytest.raises(ValueError):
        adaptive_sigma_train(1.0, 0.0)


def test_count_noise_below_half_sigma_rejected():
    # 预支公式 (σ⁻² − (2σ_c)⁻²)⁻¹/² 在 2σ_c ≤ σ 时根号内为负
    with pytest.raises(ValueError):
        adaptive_sigma_train(1.0, 0.5)


def test_anchor_sigma_equal_count_noise():
    # σ=1.0, σ_c=1.0 → (1⁻² − 2⁻²)⁻¹/² = (1 − 0.25)⁻¹/² = 2/√3
    assert abs(adaptive_sigma_train(1.0, 1.0) - 2 / 3 ** 0.5) < 1e-9


def test_monotone_in_count_noise():
    # Larger clip-count noise is itself more private (1/(2σ_c)² term shrinks),
    # so less budget pre-pay is needed: σ_train decreases toward sigma.
    s1 = adaptive_sigma_train(1.0, 1.0)
    s2 = adaptive_sigma_train(1.0, 2.0)
    assert s2 < s1
    assert s2 > 1.0  # still above the target sigma (pre-pay never vanishes)


import numpy as np
import torch

from fl_code.fed_core.client_core import train_client
from fl_code.fed_core.data import load_client_cache
from fl_code.models import TCNConfig, build_tcn


def _dp_cfg(rounds=1):
    return {"lr": 0.001, "batch_size": 16, "local_epochs": 1,
            "device": "cpu", "round": 1, "rounds": rounds,
            "budget_path": None}


def test_dp_round_reports_clip_fraction():
    cache = load_client_cache("steel_ind_0", stride=6, max_seqs=1)
    model = build_tcn(TCNConfig())
    keys = list(model.state_dict().keys())
    dp = {"mode": "uniform", "sigma": 1.0, "clipping_norm": 1.0,
          "delta": 1e-5, "target_epsilon": 0.0}
    result = train_client(
        [v.detach().numpy() for v in model.state_dict().values()],
        keys, model, cache["train_ds"], cache["n_train"], _dp_cfg(), dp)
    assert result["clip_fraction"] is not None
    assert 0.0 <= result["clip_fraction"] <= 1.0
    assert np.isfinite(result["clip_fraction"])


def test_plain_round_clip_fraction_none():
    cache = load_client_cache("steel_ind_0", stride=6, max_seqs=1)
    model = build_tcn(TCNConfig())
    keys = list(model.state_dict().keys())
    result = train_client(
        [v.detach().numpy() for v in model.state_dict().values()],
        keys, model, cache["train_ds"], cache["n_train"], _dp_cfg(), None)
    assert result["clip_fraction"] is None
