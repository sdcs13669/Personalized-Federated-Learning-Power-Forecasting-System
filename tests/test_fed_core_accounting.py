from fl_code.fed_core.accounting import dp_epsilon, sigma_for_epsilon


def test_anchor_small_client_full():
    # 与 docs/差分隐私参数推导说明.md 第 9 节表格一致
    eps = dp_epsilon(2305, 64, 1, 30, 1.0, 1e-5)
    assert abs(eps - 5.94) < 0.05, f"eps={eps}"


def test_anchor_two_rounds():
    eps = dp_epsilon(2305, 64, 1, 2, 1.0, 1e-5)
    assert abs(eps - 1.7638) < 0.05, f"eps={eps}"


def test_sigma_solver_hits_target():
    sigma, eps = sigma_for_epsilon(2312, 64, 1, 30, 1e-5, 7.5)
    assert abs(eps - 7.5) / 7.5 < 0.003, f"sigma={sigma} eps={eps}"
