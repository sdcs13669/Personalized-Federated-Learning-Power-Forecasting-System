"""flwr 客户端训练线程：加载本地采集数据 → FedClient → start_client。"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pandas as pd

from fl_code.data_utils import (make_sliding_windows, preprocess,
                                split_train_test, PowerDataset)
from fl_code.fed_core.client_core import CidEchoClient, FedClient
from fl_code.models import TCNConfig, build_tcn

# 首次连接训练通道的重试窗口。flwr 客户端「连不上就立刻退出、不重试」，
# 而服务端的 flwr worker 要等点位「开始训练」才启动 —— 谁先谁后会踩空。
# 这里给一个宽松窗口，让客户端先点也不至于整场训练缺席。
CONNECT_RETRY_SECONDS = 180
CONNECT_RETRY_INTERVAL = 3.0

_state = {"thread": None, "running": False, "round": 0,
          "loss": None, "grpc_addr": None, "error": None}


def build_train_cache(csv_path: str, seqs: list[str],
                      public_cols: list[str], local_cols: list[str],
                      input_steps: int = 144, pred_len: int = 6,
                      stride: int = 48) -> dict:
    """本地 CSV → 归一化 → 80/20 切分 → 滑窗 Dataset（同 train_baseline 数据管线）。"""
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    # 与 load_client_data 一致：category_id 展开为 one-hot 公共特征
    if "cat_residential" in public_cols and "category_id" in df.columns:
        cat = df["category_id"].astype(int)
        df["cat_residential"] = (cat == 0).astype(float)
        df["cat_transformer"] = (cat == 1).astype(float)
        df["cat_industrial"] = (cat == 2).astype(float)
        df = df.drop(columns=["category_id"])
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


class _StatusClient(CidEchoClient):
    """CidEchoClient 加状态旁路：每轮 fit 把轮次/loss 写回 trainer._state，
    供 /local/train-status（客户端页"训练中 · round=… · loss=…"）展示。
    只读 fit 入参与返回值，不改训练逻辑（fed_core 仍是唯一真源）。"""

    def fit(self, parameters, config):
        _state["round"] = int(config.get("server_round") or 0)
        # 收到第一轮指令就说明连接已建立：清掉连接期的重试告警，
        # 否则训练成功结束后界面上仍会显示"未连接训练通道"。
        _state["error"] = None
        tensors, n_train, metrics = super().fit(parameters, config)
        _state["loss"] = metrics.get("loss")
        return tensors, n_train, metrics


def _run_flwr(grpc_addr: str, cache: dict, keys: list[str], cfg: dict) -> None:
    """连接训练通道并开始联邦训练；首次连接失败会自动重试。

    为什么必须重试：flwr 客户端连不上会**立刻退出且不重试**（实测
    `Connection refused` 后进程直接结束）。而服务端的 flwr worker 要等
    「开始训练」才监听 8089 —— 若客户端先点，就会整场训练都缺席。
    更糟的是原来这个失败完全静默：`/local/start` 已返回"训练已启动"，
    异常在线程里被吞掉，界面上只看到该客户端"掉线"，无法判断原因。
    现在改为：重试到 CONNECT_RETRY_SECONDS，并把最后一次错误写进 _state。
    """
    from flwr.client import start_client
    _state["round"] = 0
    _state["loss"] = None
    _state["error"] = None
    _state["running"] = True
    deadline = time.time() + CONNECT_RETRY_SECONDS
    try:
        while True:
            # 每次尝试都新建 client，避免复用已失败连接的内部状态
            inner = FedClient(cache, keys, {**cfg, "budget_path": None})
            client = _StatusClient(inner, cfg["client_id"]).to_client()
            try:
                start_client(server_address=grpc_addr, client=client)
                return                      # 正常跑完
            except Exception as e:          # noqa: BLE001
                # 只在「一次都没参与过任何一轮」时重试；已经跑过轮次说明是
                # 中途断连，重连会让它以新身份插进训练中段，反而更乱。
                if _state["round"] > 0 or time.time() >= deadline:
                    _state["error"] = (
                        f"无法连接训练通道 {grpc_addr}：{e}。"
                        f"请确认服务端已点「开始训练」（worker 才会监听 8089），"
                        f"并检查「训练通道」地址是否为服务端实际 IP。")
                    raise
                _state["error"] = (
                    f"连接训练通道失败，自动重试中"
                    f"（剩余 {int(deadline - time.time())} 秒）：{e}")
                time.sleep(CONNECT_RETRY_INTERVAL)
    finally:
        _state["running"] = False


def start_training(grpc_addr: str, client_id: str, cfg: dict) -> str:
    """在后台线程启动 flwr 客户端；返回状态消息。"""
    if _state["thread"] is not None and _state["thread"].is_alive():
        return "训练已在运行"
    # 身份为空时原样往下走会变成 config[""] 的 KeyError，报错完全看不懂。
    # 最常见的来源是误用了默认的 run_client.bat（其 client_id 为空）。
    if not client_id:
        raise RuntimeError(
            "本客户端身份为空，无法开始训练。请使用对应机器的启动脚本"
            "（如 run_client_m2c1.bat），不要用默认的 run_client.bat。")
    csv = _find_csv()
    if csv is None:
        raise RuntimeError("未找到已采集的数据，请先采集")

    import yaml
    with open(Path(__file__).resolve().parent.parent
              / "fl_code" / "models" / "client_config.yaml") as f:
        config = yaml.safe_load(f)
    dataset_id = client_id.rsplit("_", 1)[0]
    if dataset_id not in config:
        raise RuntimeError(
            f"client_id {client_id!r} 与数据集配置不匹配（推导出 {dataset_id!r}）。"
            f"可用数据集：{', '.join(config.keys())}")
    dcfg = config[dataset_id]
    client_cfg = dcfg["clients"].get(client_id)
    if client_cfg is None:
        raise RuntimeError(f"client_config 中找不到客户端: {client_id}")
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
    data_dir = Path(os.environ.get("FL_DATA_DIR",
                                   str(Path(__file__).resolve().parent / "data")))
    if not data_dir.exists():
        return None
    csvs = list(data_dir.glob("*.csv"))
    return csvs[0] if csvs else None


def get_train_status() -> dict:
    thread = _state["thread"]
    return {"running": _state["running"],
            "alive": thread is not None and thread.is_alive(),
            "round": _state["round"], "loss": _state["loss"],
            "error": _state["error"],
            "grpc_addr": _state["grpc_addr"]}
