import numpy as np

from fl_code.plot_results import _stage_table


def _baseline(wape, avg_mae, clients):
    return {"final_metrics": {"wape": wape, "avg_mae": avg_mae,
                              "client_metrics": clients}}


def test_stage_table_extracts_three_stages():
    nodp = _baseline(0.29, 0.23, {"c0": {"mae": 0.2, "rmse": 0.3, "r2": 0.9}})
    dp = _baseline(0.42, 0.36, {"c0": {"mae": 0.3, "rmse": 0.4, "r2": 0.8},
                                "c1": {"mae": 0.5, "rmse": 0.6, "r2": 0.7}})
    pers = {"final_metrics": {
        "avg_mae_personalized": 0.33,
        "wape_personalized": 0.40,
        "client_metrics": {"c0": {"mae": 0.18, "rmse": 0.28, "r2": 0.91}},
    }}
    t = _stage_table(nodp, dp, pers)
    assert t["nodp"]["wape"] == 0.29 and t["nodp"]["avg_mae"] == 0.23
    assert t["dp"]["wape"] == 0.42 and t["dp"]["avg_mae"] == 0.36
    assert t["dp+rc"]["wape"] == 0.40 and t["dp+rc"]["avg_mae"] == 0.33
    assert set(t["dp"]["clients"]) == {"c0", "c1"}
    assert t["nodp"]["clients"]["c0"]["mae"] == 0.2


def test_stage_table_missing_stages_and_keys_are_nan():
    t = _stage_table(None, None, None)
    for s in ("nodp", "dp", "dp+rc"):
        assert np.isnan(t[s]["wape"]) and np.isnan(t[s]["avg_mae"])
        assert t[s]["clients"] == {}
    # 旧版 personalized_results.json 没有 wape_personalized 键
    t = _stage_table(None, None, {"final_metrics": {"avg_mae_personalized": 0.3}})
    assert np.isnan(t["dp+rc"]["wape"])
    assert t["dp+rc"]["avg_mae"] == 0.3


def test_stage_table_client_union_sorted():
    nodp = _baseline(0.1, 0.1, {"b": {"mae": 1, "rmse": 1}})
    dp = _baseline(0.1, 0.1, {"a": {"mae": 1, "rmse": 1}})
    t = _stage_table(nodp, dp, None)
    assert sorted(t["nodp"]["clients"]) == ["b"]
    assert sorted(t["dp"]["clients"]) == ["a"]
