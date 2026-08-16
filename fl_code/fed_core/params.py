"""State-dict <-> flwr parameter tensor conversion."""
from __future__ import annotations

import numpy as np
import torch


def state_dict_to_tensors(state: dict[str, torch.Tensor]) -> list[np.ndarray]:
    # .copy(): the numpy array must not alias the tensor's memory (the
    # caller may mutate it, e.g. DP noise added in place).
    return [v.detach().cpu().numpy().copy() for v in state.values()]


def tensors_to_state_dict(tensors: list[np.ndarray],
                          keys: list[str]) -> dict[str, torch.Tensor]:
    if len(tensors) != len(keys):
        raise ValueError(
            f"tensors/keys 长度不匹配: {len(tensors)} != {len(keys)}")
    return {k: torch.from_numpy(np.asarray(t, dtype=np.float32))
            for k, t in zip(keys, tensors)}
