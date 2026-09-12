"""app/trainer.start_training 的前置校验。

把「看不懂的 KeyError」变成可操作的中文提示。最典型的来源是误用默认的
run_client.bat —— 它的 client_id 是空的，会一路走到 config[""] 抛 KeyError。
"""
import pytest

from app import trainer

CFG = {"batch_size": 64, "local_epochs": 1, "lr": 0.001}


def test_empty_client_id_gives_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setenv("FL_DATA_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="身份为空"):
        trainer.start_training("127.0.0.1:8089", "", CFG)


def test_missing_collected_data_gives_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setenv("FL_DATA_DIR", str(tmp_path))     # 空目录，没有 csv
    with pytest.raises(RuntimeError, match="未找到已采集的数据"):
        trainer.start_training("127.0.0.1:8089", "steel_ind_0", CFG)


def test_unknown_dataset_gives_actionable_error(monkeypatch, tmp_path):
    (tmp_path / "x.csv").write_text("datetime,a\n", encoding="utf-8")
    monkeypatch.setenv("FL_DATA_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="与数据集配置不匹配"):
        trainer.start_training("127.0.0.1:8089", "nosuch_0", CFG)


def test_unknown_client_id_gives_actionable_error(monkeypatch, tmp_path):
    (tmp_path / "x.csv").write_text("datetime,a\n", encoding="utf-8")
    monkeypatch.setenv("FL_DATA_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="找不到客户端"):
        trainer.start_training("127.0.0.1:8089", "steel_ind_9", CFG)


def test_all_six_demo_client_ids_resolve(monkeypatch, tmp_path):
    """演示用的 6 个 client_id 必须都能在 client_config.yaml 里对应到数据集。"""
    import yaml
    from pathlib import Path
    cfg_path = (Path(trainer.__file__).resolve().parent.parent
                / "fl_code" / "models" / "client_config.yaml")
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    for cid in ("steel_ind_0", "tetouan_city_0", "tetouan_city_1",
                "tetouan_city_2", "lcl_res_0", "lcl_res_1"):
        ds = cid.rsplit("_", 1)[0]
        assert ds in config, f"{cid} 推导出的数据集 {ds} 不在配置里"
        assert cid in config[ds]["clients"], f"{cid} 不在 {ds} 的 clients 里"
