"""PLD privacy accounting for client-side DP-SGD (Google dp-accounting).

Moved verbatim from train_baseline.py; numeric anchors (see tests) must
stay identical.
"""
from __future__ import annotations

from math import ceil


def dp_epsilon(n_train: int, batch_size: int, local_epochs: int,
               rounds: int, noise_multiplier: float, delta: float) -> float:
    from dp_accounting import dp_event
    from dp_accounting.pld import pld_privacy_accountant

    q = batch_size / n_train
    steps = ceil(n_train / batch_size) * local_epochs * rounds
    accountant = pld_privacy_accountant.PLDAccountant()
    accountant.compose(dp_event.PoissonSampledDpEvent(
        q, dp_event.GaussianDpEvent(noise_multiplier)), steps)
    return float(accountant.get_epsilon(delta))


def dp_epsilon_worst(client_sizes: list[int], batch_size: int,
                     local_epochs: int, rounds: int,
                     noise_multiplier: float, delta: float) -> float:
    return max(dp_epsilon(n, batch_size, local_epochs, rounds,
                          noise_multiplier, delta) for n in client_sizes)


def sigma_for_epsilon(n_train: int, batch_size: int, local_epochs: int,
                      rounds: int, delta: float, target: float
                      ) -> tuple[float, float]:
    from math import exp, log

    def eps_at(sigma: float) -> float:
        return dp_epsilon(n_train, batch_size, local_epochs, rounds,
                          sigma, delta)

    sigma_a, eps_a = 1.0, eps_at(1.0)
    if abs(eps_a - target) / target < 0.002:
        return sigma_a, eps_a
    sigma_b = sigma_a * (eps_a / target) ** 0.5
    eps_b = eps_at(sigma_b)
    for _ in range(6):
        if abs(eps_b - target) / target < 0.002:
            break
        xa, ya = log(sigma_a), log(eps_a)
        xb, yb = log(sigma_b), log(eps_b)
        sigma_c = exp(xa + (log(target) - ya) * (xb - xa) / (yb - ya))
        sigma_a, eps_a = sigma_b, eps_b
        sigma_b, eps_b = sigma_c, eps_at(sigma_c)
    return sigma_b, eps_b


def adaptive_sigma_train(sigma: float, clip_count_noise: float) -> float:
    """Budget pre-pay: training noise multiplier that covers the noised
    clip-count mechanism (Andrew et al. 2021, Thm 1; Flower DPFedAvgAdaptive).

    Accounting still reports ε at ``sigma``; training actually uses the
    larger returned value.  Requires ``clip_count_noise > sigma / 2``
    (the pre-pay formula is undefined otherwise).
    """
    if clip_count_noise <= 0:
        raise ValueError("clip_count_noise must be > 0")
    if 2 * clip_count_noise <= sigma:
        raise ValueError(
            f"clip_count_noise {clip_count_noise} must be > sigma/2 "
            f"({sigma / 2})")
    return float((sigma ** -2 - (2 * clip_count_noise) ** -2) ** -0.5)
