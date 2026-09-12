"""app/client_stage 二阶段结果解析：键名正确性 + stdout 抓取健壮性。

两个真实踩过的坑（2026-09-12 实测确认）：
  1. train_personalized 把每客户端结果放在顶层 "results" 下，而 client_stage
     曾误读 "client_metrics" → cid_data 恒为空 → wape_global/wape_rc 永远 None。
  2. subprocess 用 text=True 且未指定 encoding，一旦子进程输出 UTF-8（例如外层
     设了 PYTHONIOENCODING=utf-8），stdout 会变成 None，随后
     _parse_epoch_losses(None) 抛 AttributeError，二阶段直接失败。
"""
import json

from app.client_stage import (_child_env, _parse_epoch_losses, _pct,
                              _read_client_metrics)


def _write(tmp_path, payload):
    p = tmp_path / "personalized_results.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _read_client_metrics：键名
# ---------------------------------------------------------------------------

def test_reads_per_client_result_from_results_key(tmp_path):
    """train_personalized 的真实结构：per-client 结果在顶层 results 下。"""
    per = {"wape_baseline": 0.9859, "wape_personalized": 0.8605,
           "epoch_losses": [0.278277, 0.241518], "best_epoch": 2}
    p = _write(tmp_path, {
        "args": {}, "global_model": "x.pt", "num_clients": 1,
        "final_metrics": {}, "dp": None,
        "results": {"steel_ind_0": per},
        # 真实文件里【没有】client_metrics 键
    })

    got = _read_client_metrics(p, "steel_ind_0")

    assert got == per
    assert got["wape_baseline"] == 0.9859
    assert got["wape_personalized"] == 0.8605


def test_falls_back_to_client_metrics_key(tmp_path):
    """兼容早期可能存在的 client_metrics 结构。"""
    per = {"wape_baseline": 1.0, "wape_personalized": 0.5}
    p = _write(tmp_path, {"client_metrics": {"c1": per}})

    assert _read_client_metrics(p, "c1") == per


def test_prefers_results_over_client_metrics(tmp_path):
    p = _write(tmp_path, {"results": {"c1": {"wape_rc": 2.0}},
                          "client_metrics": {"c1": {"wape_rc": 9.9}}})

    assert _read_client_metrics(p, "c1")["wape_rc"] == 2.0


def test_missing_file_returns_empty(tmp_path):
    assert _read_client_metrics(tmp_path / "nope.json", "c1") == {}


def test_unknown_client_returns_empty(tmp_path):
    p = _write(tmp_path, {"results": {"other": {"a": 1}}})
    assert _read_client_metrics(p, "c1") == {}


def test_broken_json_returns_empty(tmp_path):
    p = tmp_path / "personalized_results.json"
    p.write_text("{not valid json", encoding="utf-8")
    assert _read_client_metrics(p, "c1") == {}


def test_empty_results_returns_empty(tmp_path):
    p = _write(tmp_path, {"results": {}})
    assert _read_client_metrics(p, "c1") == {}


# ---------------------------------------------------------------------------
# _parse_epoch_losses：健壮性
# ---------------------------------------------------------------------------

def test_parse_epoch_losses_primary_pattern():
    out = _parse_epoch_losses("EPOCHLOSS 1 0.278277\nEPOCHLOSS 2 0.241518\n")
    assert out == [0.278277, 0.241518]


def test_parse_epoch_losses_handles_none_stdout():
    """stdout 抓取失败为 None —— 早期这里会 AttributeError 让二阶段整条崩掉。"""
    assert _parse_epoch_losses(None) == []


def test_parse_epoch_losses_handles_empty():
    assert _parse_epoch_losses("") == []


def test_parse_epoch_losses_fallback_pattern():
    out = _parse_epoch_losses("Epoch 1/5  loss=0.123456  1.2s\n"
                              "Epoch 2/5  loss=0.100000  1.1s\n")
    assert out == [0.123456, 0.1]


def test_parse_epoch_losses_ignores_garbage_lines():
    out = _parse_epoch_losses("some log\nEPOCHLOSS bad\nEPOCHLOSS 3 0.5\n")
    assert out == [0.5]


# ---------------------------------------------------------------------------
# _child_env：钉死 UTF-8
# ---------------------------------------------------------------------------

def test_child_env_forces_utf8_io():
    env = _child_env()
    assert env["PYTHONIOENCODING"] == "utf-8"


# ---------------------------------------------------------------------------
# _pct：WAPE 单位统一（小数 -> 百分比）
# ---------------------------------------------------------------------------

def test_pct_converts_fraction_to_percent():
    """train_personalized 给的是小数，项目其它地方用百分比。"""
    assert _pct(0.0524) == 5.24
    assert _pct(0.985947847366333) == 98.5948


def test_pct_handles_none_and_zero():
    assert _pct(None) is None
    assert _pct(0) == 0.0
    assert _pct(0.0) == 0.0


def test_pct_handles_bad_type():
    assert _pct("not-a-number") is None
    assert _pct([1, 2]) is None


def test_pct_on_real_result_payload(tmp_path):
    """用真实 personalized_results.json 的形状串一遍：小数 -> 百分比。"""
    per = {"wape_baseline": 0.985947847366333,
           "wape_personalized": 0.8605257868766785,
           "epoch_losses": [0.278277, 0.241518]}
    p = tmp_path / "personalized_results.json"
    p.write_text(json.dumps({"results": {"steel_ind_0": per}}), encoding="utf-8")

    got = _read_client_metrics(p, "steel_ind_0")
    assert _pct(got["wape_baseline"]) == 98.5948
    assert _pct(got["wape_personalized"]) == 86.0526
