import numpy as np
import torch

import fl_code.visualize_eval as ve
from fl_code.visualize_eval import _metrics


def _make_ckpt(root, sub):
    d = root / sub / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"w": torch.zeros(1)}, d / "round_001.pt")


def _make_corrector(pers, eps, rc_type, cid="steel_ind_0"):
    d = pers / f"epsilon-{eps}" / rc_type
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"w": torch.zeros(1)}, d / f"corrector_{cid}.pt")


def test_wape_in_metrics():
    a = np.array([10.0, 20.0, 30.0])
    p = np.array([11.0, 18.0, 33.0])
    m = _metrics(a, p)
    assert abs(m["wape"] - 6.0 / 60.0) < 1e-12
    assert abs(m["mae"] - 2.0) < 1e-12


def test_wape_nan_when_actuals_all_zero():
    m = _metrics(np.zeros(3), np.array([1.0, 2.0, 3.0]))
    assert np.isnan(m["wape"])


def test_wape_ignores_nan_pairs():
    a = np.array([10.0, np.nan, 30.0])
    p = np.array([12.0, 99.0, 27.0])
    m = _metrics(a, p)
    assert abs(m["wape"] - 5.0 / 40.0) < 1e-12


def test_model_menu_three_kinds(tmp_path, monkeypatch):
    _make_ckpt(tmp_path, "nodp")
    _make_ckpt(tmp_path, "dp/epsilon-5.5")
    _make_ckpt(tmp_path, "dp/epsilon-7.5")
    pers = tmp_path / "pers"
    _make_corrector(pers, "5.5", "mlp")
    monkeypatch.setattr(ve, "PERSONALIZED_DIR", pers)
    det = ve.EvalVisualizer._detect_models(None, "steel_ind_0", tmp_path)
    assert det["nodp"] is not None
    assert set(det["dp"]) == {"5.5", "7.5"}
    assert det["dp+rc"] == {
        "5.5": {"mlp": pers / "epsilon-5.5" / "mlp"
                / "corrector_steel_ind_0.pt"}}


def test_model_menu_no_rc_without_corrector(tmp_path, monkeypatch):
    _make_ckpt(tmp_path, "nodp")
    _make_ckpt(tmp_path, "dp/epsilon-5.5")
    pers = tmp_path / "pers"
    pers.mkdir()
    monkeypatch.setattr(ve, "PERSONALIZED_DIR", pers)
    det = ve.EvalVisualizer._detect_models(None, "steel_ind_0", tmp_path)
    assert det["nodp"] is not None and set(det["dp"]) == {"5.5"}
    assert det["dp+rc"] == {}


def test_model_menu_rc_requires_dp_checkpoint(tmp_path, monkeypatch):
    _make_ckpt(tmp_path, "nodp")  # dp 无 checkpoint
    pers = tmp_path / "pers"
    _make_corrector(pers, "5.5", "mlp")
    monkeypatch.setattr(ve, "PERSONALIZED_DIR", pers)
    det = ve.EvalVisualizer._detect_models(None, "steel_ind_0", tmp_path)
    assert det["nodp"] is not None and det["dp"] == {}
    assert det["dp+rc"] == {}


def test_model_menu_corrector_epsilon_must_match_dp(tmp_path, monkeypatch):
    _make_ckpt(tmp_path, "dp/epsilon-5.5")
    pers = tmp_path / "pers"
    _make_corrector(pers, "7.5", "mlp")  # corrector 在 ε=7.5，dp global 只有 5.5
    monkeypatch.setattr(ve, "PERSONALIZED_DIR", pers)
    det = ve.EvalVisualizer._detect_models(None, "steel_ind_0", tmp_path)
    assert set(det["dp"]) == {"5.5"}
    assert det["dp+rc"] == {}


def test_model_menu_skips_empty_epsilon_dir(tmp_path):
    _make_ckpt(tmp_path, "dp/epsilon-5.5")
    (tmp_path / "dp" / "epsilon-0.5").mkdir(parents=True)  # 空目录
    det = ve.EvalVisualizer._detect_models(None, "steel_ind_0", tmp_path)
    assert set(det["dp"]) == {"5.5"}
