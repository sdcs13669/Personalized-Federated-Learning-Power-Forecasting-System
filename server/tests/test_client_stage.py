"""app/client_stage 二阶段结果解析：键名正确性 + stdout 抓取健壮性。

两个真实踩过的坑（2026-09-12 实测确认）：
  1. train_personalized 把每客户端结果放在顶层 "results" 下，而 client_stage
     曾误读 "client_metrics" → cid_data 恒为空 → wape_global/wape_rc 永远 None。
  2. subprocess 用 text=True 且未指定 encoding，一旦子进程输出 UTF-8（例如外层
     设了 PYTHONIOENCODING=utf-8），stdout 会变成 None，随后
     _parse_epoch_losses(None) 抛 AttributeError，二阶段直接失败。
"""
import json
from pathlib import Path

from app.client_stage import (_child_env, _parse_epoch_losses, _pct,
                              _read_client_metrics)
import app.client_stage as client_stage


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


# ---------------------------------------------------------------------------
# run_stage2 的命令构造：epochs / stride 可按客户端配置
#
# 背景：窗口数随「序列数」成倍增长（lcl_res 有 5~6 条序列 → 13 倍窗口），
# 默认 15 epoch 在 lcl_res 上要跑 30~40 分钟。这两个参数让大客户端可按配置
# 调大 stride、减少 epoch，把耗时拉回 3~4 分钟量级。
# ---------------------------------------------------------------------------

class _FakeProc:
    returncode = 0
    stdout = "EPOCHLOSS 1 0.5\nEPOCHLOSS 2 0.4\n"
    stderr = ""


def _patch_stage2(tmp_path, monkeypatch):
    """隔离目录 + 假下载 + 假子进程，返回被调用命令的列表。"""
    monkeypatch.setattr(client_stage, "STAGE_DIR", tmp_path / "stage")
    monkeypatch.setattr(client_stage, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(client_stage, "_download_global_model",
                        lambda *a, **k: tmp_path / "global.pt")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "x.csv").write_text("datetime,a\n", encoding="utf-8")

    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(list(cmd))
        out = cmd[cmd.index("--output-dir") + 1]
        rundir = Path(out) / "tcn"
        rundir.mkdir(parents=True, exist_ok=True)
        (rundir / "personalized_results.json").write_text(json.dumps({
            "results": {"steel_ind_0": {
                "wape_baseline": 0.05, "wape_personalized": 0.04,
                "epoch_losses": [0.5, 0.4]}}}), encoding="utf-8")
        return _FakeProc()

    monkeypatch.setattr(client_stage.subprocess, "run", fake_run)
    return seen, str(data_dir)


def test_stage2_default_command_uses_fl_code_defaults(tmp_path, monkeypatch):
    seen, data_dir = _patch_stage2(tmp_path, monkeypatch)
    client_stage.run_stage2("http://s", "tok", 7, "steel_ind_0", "tcn", data_dir)

    argv = seen[-1]
    assert argv[argv.index("--epochs") + 1] == "15"
    assert "--stride" not in argv, "默认不应显式传 stride（沿用 fl_code 默认）"
    assert argv[argv.index("--rc-type") + 1] == "tcn"
    assert argv[argv.index("--clients") + 1] == "steel_ind_0"


def test_stage2_honours_configured_epochs_and_stride(tmp_path, monkeypatch):
    seen, data_dir = _patch_stage2(tmp_path, monkeypatch)
    client_stage.run_stage2("http://s", "tok", 7, "lcl_res_0", "tcn", data_dir,
                            epochs=6, stride=192)

    argv = seen[-1]
    assert argv[argv.index("--epochs") + 1] == "6"
    assert argv[argv.index("--stride") + 1] == "192"


def test_stage2_epochs_only(tmp_path, monkeypatch):
    seen, data_dir = _patch_stage2(tmp_path, monkeypatch)
    client_stage.run_stage2("http://s", "tok", 7, "lcl_res_0", "tcn", data_dir,
                            epochs=3)

    argv = seen[-1]
    assert argv[argv.index("--epochs") + 1] == "3"
    assert "--stride" not in argv


def test_stage2_state_marks_done_with_percent_wape(tmp_path, monkeypatch):
    _seen, data_dir = _patch_stage2(tmp_path, monkeypatch)
    st = client_stage.run_stage2("http://s", "tok", 7, "steel_ind_0", "tcn",
                                 data_dir)

    assert st["stage2"] == "done"
    assert st["wape_global"] == 5.0        # 0.05 -> 5%
    assert st["wape_rc"] == 4.0            # 0.04 -> 4%
    assert st["epoch_losses"] == [0.5, 0.4]
