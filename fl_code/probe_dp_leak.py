"""临时诊断脚本：隔离 _train_dp / _train_plain 是否有 CUDA 内存泄漏。

在 GPU 服务器上运行（不写任何输出文件，不动正式产物）:

    conda activate fl && python fl_code/probe_dp_leak.py

先跑 3 轮 plain（对照），再跑 8 轮 DP。逐轮打印 allocated/reserved。
判读: 若 DP 的 allocated 逐轮单调上涨 => 泄漏在 _train_dp 内部
      若 DP 平线而 plain 也平线  => 泄漏在 flwr/Ray 外围，不在训练循环
"""
import argparse
import gc

import torch

from fl_code.fed_core.client_core import _train_dp, _train_plain
from fl_code.fed_core.data import load_client_cache
from fl_code.models import TCNConfig, build_tcn


def mem_mb() -> str:
    if not torch.cuda.is_available():
        return "cuda N/A"
    a = torch.cuda.memory_allocated() / 1e6
    r = torch.cuda.memory_reserved() / 1e6
    return f"alloc={a:7.1f}MB reserved={r:7.1f}MB"


def run(mode: str, train_ds, rounds: int, device: str) -> None:
    print(f"--- {mode} x{rounds} rounds ---")
    for r in range(rounds):
        model = build_tcn(TCNConfig()).to(device)
        if mode == "dp":
            loss = _train_dp(model, train_ds, lr=1e-3, batch_size=16,
                             local_epochs=1, device=device,
                             noise_multiplier=1.0, clipping_norm=1.0)
        else:
            loss = _train_plain(model, train_ds, lr=1e-3, batch_size=16,
                                local_epochs=1, device=device)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"round {r:2d}: loss={loss:.4f}  {mem_mb()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8, help="DP rounds")
    ap.add_argument("--plain-rounds", type=int, default=3)
    ap.add_argument("--client", default="steel_ind_0")
    ap.add_argument("--max-seqs", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__} client={args.client} "
          f"max_seqs={args.max_seqs}")
    cache = load_client_cache(args.client, stride=6, max_seqs=args.max_seqs)
    train_ds = cache["train_ds"]
    print(f"n_train={cache['n_train']}  start: {mem_mb()}")

    run("plain", train_ds, args.plain_rounds, device)
    run("dp", train_ds, args.rounds, device)
    print("done")


if __name__ == "__main__":
    main()
