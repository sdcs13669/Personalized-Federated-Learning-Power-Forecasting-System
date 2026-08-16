"""State-dict <-> flwr parameter tensor conversion."""
from __future__ import annotations

import numpy as np
import torch


def state_dict_to_tensors(state: dict[str, torch.Tensor]) -> list[np.ndarray]:
    return [v.detach().cpu().numpy() for v in state.values()]


def tensors_to_state_dict(tensors: list[np.ndarray],
                          keys: list[str]) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.asarray(t, dtype=np.float32))
            for k, t in zip(keys, tensors)}
