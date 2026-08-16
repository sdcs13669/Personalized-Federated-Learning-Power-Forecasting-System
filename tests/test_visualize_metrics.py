import numpy as np
import torch

import fl_code.visualize_eval as ve
from fl_code.visualize_eval import _metrics


def _make_ckpt(root, sub):
    d = root / sub / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"w": torch.zeros(1)}, d / "round_001.pt")


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
    _make_ckpt(tmp_path, "dp")
    pers = tmp_path / "pers"
    pers.mkdir()
    torch.save({"w": torch.zeros(1)}, pers / "corrector_steel_ind_0.pt")
    monkeypatch.setattr(ve, "PERSONALIZED_DIR", pers)
    opts = ve.EvalVisualizer._detect_models(None, "steel_ind_0", tmp_path)
    assert [label for label, _, _ in opts] == ["nodp", "dp", "dp+rc"]


def test_model_menu_no_rc_without_corrector(tmp_path, monkeypatch):
    _make_ckpt(tmp_path, "nodp")
    _make_ckpt(tmp_path, "dp")
    pers = tmp_path / "pers"
    pers.mkdir()
    monkeypatch.setattr(ve, "PERSONALIZED_DIR", pers)
    opts = ve.EvalVisualizer._detect_models(None, "steel_ind_0", tmp_path)
    assert [label for label, _, _ in opts] == ["nodp", "dp"]


def test_model_menu_rc_requires_dp_checkpoint(tmp_path, monkeypatch):
    _make_ckpt(tmp_path, "nodp")  # dp 无 checkpoint
    pers = tmp_path / "pers"
    pers.mkdir()
    torch.save({"w": torch.zeros(1)}, pers / "corrector_steel_ind_0.pt")
    monkeypatch.setattr(ve, "PERSONALIZED_DIR", pers)
    opts = ve.EvalVisualizer._detect_models(None, "steel_ind_0", tmp_path)
    assert [label for label, _, _ in opts] == ["nodp"]
