"""flwr 客户端训练线程：加载本地采集数据 → FedClient → start_client。"""
from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd

from fl_code.data_utils import (make_sliding_windows, preprocess,
                                split_train_test, PowerDataset)
from fl_code.fed_core.client_core import CidEchoClient, FedClient
from fl_code.models import TCNConfig, build_tcn

_state = {"thread": None, "running": False, "round": 0,
          "loss": None, "grpc_addr": None}


def build_train_cache(csv_path: str, seqs: list[str],
                      public_cols: list[str], local_cols: list[str],
                      input_steps: int = 144, pred_len: int = 6,
                      stride: int = 48) -> dict:
    """本地 CSV → 归一化 → 80/20 切分 → 滑窗 Dataset（同 train_baseline 数据管线）。"""
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df_norm, _ = preprocess(df, seqs, local_cols)
    train_df, test_df = split_train_test(df_norm, seqs)
    X, y, X_local, _ = make_sliding_windows(
        train_df, seqs, public_cols, input_steps=input_steps,
        pred_len=pred_len, stride=stride, train=True, local_cols=local_cols)
    Xt, yt, _, _ = make_sliding_windows(
        test_df, seqs, public_cols, input_steps=input_steps,
        pred_len=pred_len, stride=stride, train=False, local_cols=local_cols)
    return {
        "train_ds": PowerDataset(X, y, X_local),
        "test_ds": PowerDataset(Xt, yt, None),
        "n_train": int(len(X)),
    }


def _state_dict_keys() -> list[str]:
    return list(build_tcn(TCNConfig()).state_dict().keys())


def _run_flwr(grpc_addr: str, cache: dict, keys: list[str], cfg: dict) -> None:
    from flwr.client import start_client
    inner = FedClient(cache, keys, {**cfg, "budget_path": None})
    client = CidEchoClient(inner, cfg["client_id"]).to_client()
    _state["running"] = True
    try:
        start_client(server_address=grpc_addr, client=client)
    finally:
        _state["running"] = False


def start_training(grpc_addr: str, client_id: str, cfg: dict) -> str:
    """在后台线程启动 flwr 客户端；返回状态消息。"""
    if _state["thread"] is not None and _state["thread"].is_alive():
        return "训练已在运行"
    csv = _find_csv()
    if csv is None:
        raise RuntimeError("未找到已采集的数据，请先采集")

    import yaml
    with open(Path(__file__).resolve().parent.parent
              / "fl_code" / "models" / "client_config.yaml") as f:
        config = yaml.safe_load(f)
    dataset_id = client_id.rsplit("_", 1)[0]
    dcfg = config[dataset_id]
    client_cfg = next(c for c in dcfg["clients"] if c["id"] == client_id)
    seqs = client_cfg["sequences"]
    public_cols = list(dcfg["public_features"])
    local_cols = list(dcfg.get("local_features", []))
    cache = build_train_cache(str(csv), seqs, public_cols, local_cols)
    keys = _state_dict_keys()
    thread_cfg = {**cfg, "client_id": client_id}
    t = threading.Thread(target=_run_flwr, args=(grpc_addr, cache, keys,
                                                 thread_cfg), daemon=True)
    _state["thread"] = t
    t.start()
    return "训练已启动"


def _find_csv() -> Path | None:
    data_dir = Path(__file__).resolve().parent / "data"
    if not data_dir.exists():
        return None
    csvs = list(data_dir.glob("*.csv"))
    return csvs[0] if csvs else None


def get_train_status() -> dict:
    thread = _state["thread"]
    return {"running": _state["running"],
            "alive": thread is not None and thread.is_alive(),
            "round": _state["round"], "loss": _state["loss"],
            "grpc_addr": _state["grpc_addr"]}
