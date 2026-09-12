"""自适应裁剪的计数噪声保底（safe_count_noise）。

回归的坑（2026-09-12 实测）：
任务勾选「自适应裁剪」后，**所有客户端第一轮 fit 全部抛错**：
    ValueError: clip_count_noise 0.5 must be > sigma/2 (0.5199424406983865)
（client_core.train_client → accounting.adaptive_sigma_train 的契约要求）
→ 全员掉线、训练没有任何结果，而界面上只显示"掉线"，极难定位。

默认 dp_clip_count_noise=0.5，而 ε=7.5 / 12 轮时 σ≈1.47 → 必然触发。
"""
import pytest

from fl_code.fed_core.accounting import (adaptive_sigma_train,
                                         sigma_for_epsilon)
from fl_code.fed_core.client_core import safe_count_noise


def test_raises_without_clamp_documented():
    """记录原始契约：不保底时确实会抛（这里直接验 accounting 的行为）。"""
    sigma, _ = sigma_for_epsilon(231, 64, 1, 12, 1e-5, 7.5)
    assert sigma > 1.0
    with pytest.raises(ValueError):
        adaptive_sigma_train(sigma, 0.5)


def test_clamped_value_satisfies_contract():
    """保底后必须不再抛异常，且预付款系数在合理范围（<1.3σ）。"""
    sigma, _ = sigma_for_epsilon(231, 64, 1, 12, 1e-5, 7.5)
    c = safe_count_noise(0.5, sigma, adaptive=True)
    assert c > sigma / 2

    sigma_train = adaptive_sigma_train(sigma, c)
    assert sigma_train >= sigma                      # 训练噪声不得低于记账值
    assert sigma_train < 1.3 * sigma                 # 也不应过度加噪


def test_non_adaptive_keeps_value_untouched():
    sigma, _ = sigma_for_epsilon(231, 64, 1, 12, 1e-5, 7.5)
    assert safe_count_noise(0.5, sigma, adaptive=False) == 0.5


def test_large_enough_value_is_kept():
    """已经满足契约时不要改动用户配置。"""
    assert safe_count_noise(5.0, 1.47, adaptive=True) == 5.0


def test_none_and_zero_are_safe():
    assert safe_count_noise(None, 1.47, adaptive=True) == 1.47
    assert safe_count_noise(0.0, 1.47, adaptive=True) == 1.47
    # σ 未知（0）时不做处理，避免引入无意义的值
    assert safe_count_noise(0.5, 0.0, adaptive=True) == 0.5


def test_works_for_whole_epsilon_sweep():
    """ε 扫描的每一档都不能再崩。"""
    for eps in (0.5, 1.5, 2.5, 3.5, 5.5, 7.5):
        sigma, _ = sigma_for_epsilon(231, 64, 1, 12, 1e-5, eps)
        c = safe_count_noise(0.5, sigma, adaptive=True)
        sigma_train = adaptive_sigma_train(sigma, c)   # 不应抛
        assert sigma_train >= sigma
