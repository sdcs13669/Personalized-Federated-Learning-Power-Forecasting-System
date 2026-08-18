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
