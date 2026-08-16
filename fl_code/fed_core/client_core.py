"""Client-side training + DP (real implementation lands in Task 7).

Frozen contract — the App line depends only on these signatures.
"""
from __future__ import annotations


def train_client(tensors: list, keys: list[str], model, train_ds, n_train: int,
                 cfg: dict, dp: dict | None) -> dict:
    """Run one round of local (DP-)SGD from ``tensors`` (model mutated).

    cfg: {"lr", "batch_size", "local_epochs", "device", "round",
          "rounds", "budget_path": Path|None}
    dp: None | {"noise_multiplier"|None, "clipping_norm", "delta",
                "mode": "uniform"|"per_client", "sigma"|None,
                "target_epsilon"|None}
    Returns: {"tensors": list[np.ndarray], "n_train": int,
              "eps": float|None, "sigma": float|None, "loss": float}
    """
    raise NotImplementedError("Task 7")


class FedClient:
    """flwr NumPyClient wrapper (Task 7)."""

    def __init__(self, cache: dict, state_keys: list[str], cfg: dict):
        raise NotImplementedError("Task 7")

    def to_client(self):
        raise NotImplementedError("Task 7")
