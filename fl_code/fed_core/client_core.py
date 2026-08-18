"""Client-side training + DP-SGD + PLD accounting (flwr NumPyClient)."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from flwr.client import NumPyClient
from torch.utils.data import DataLoader

from fl_code.fed_core.accounting import (
    dp_epsilon, sigma_for_epsilon, adaptive_sigma_train,
)
from fl_code.fed_core.params import state_dict_to_tensors, tensors_to_state_dict
from fl_code.models import TCNConfig, build_tcn
from fl_code.train_eval_utils import train_epoch
from torch.func import functional_call, grad, vmap


def _loss_fn_dp(p, x, yy, m):
    """Stateless loss for DP per-sample grad.

    ``model`` is passed as an argument (in_dims=None) so the closure
    holds no module reference. The cross-round leak is governed by the
    params-dict identity cache in vmap (see ``_get_fp_params``), not by
    this closure.
    """
    out = functional_call(m, p, (x.unsqueeze(0),))
    return (out.squeeze(0) - yy).abs().mean()


_per_sample_grad_dp = vmap(grad(_loss_fn_dp, argnums=0),
                           in_dims=(None, 0, 0, None),
                           randomness="different")

# torch.func.vmap caches the backward graph keyed on the params dict's
# tensor identity. FedClient.fit builds a fresh model every round, so a
# fresh params dict would create a fresh cache entry every round that
# gc.collect() cannot reclaim (measured ~18 MB/round on the 64ch TCN).
# Keep ONE persistent container and copy current weights into it before
# each vmap call; the cache is then built exactly once per process.
_fp_params: dict[str, torch.Tensor] | None = None


def _get_fp_params(model) -> dict[str, torch.Tensor]:
    """Return the process-wide persistent container synced to ``model``."""
    global _fp_params
    if _fp_params is None:
        _fp_params = {n: torch.empty_like(p, device=p.device)
                      for n, p in model.named_parameters()}
    with torch.no_grad():
        for n, p in model.named_parameters():
            _fp_params[n].copy_(p)
    return _fp_params


def _train_plain(model, train_ds, lr, batch_size, local_epochs, device) -> float:
    # Parity with train_baseline._train_client: ONE DataLoader + ONE Adam
    # optimizer created OUTSIDE the local_epochs loop (Adam state/momentum
    # persists across epochs).
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = float("nan")
    for _ in range(local_epochs):
        loss = train_epoch(model, loader, optimizer, device)
    return loss


def _train_dp(model, train_ds, lr, batch_size, local_epochs, device,
              noise_multiplier: float, clipping_norm: float) -> tuple[float, float]:
    # One process-wide params container so vmap's per-dict graph cache is
    # created once; synced to the model's current weights before each call.
    params = _get_fp_params(model)

    # Parity with train_baseline._train_client_dp: the optimizer is created
    # ONCE outside the epochs loop, while the DataLoader is recreated per
    # epoch INSIDE the loop.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = float("nan")
    n_unclipped = 0
    n_samples = 0
    for _ in range(local_epochs):
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            drop_last=False)
        epoch_loss, epoch_samples = 0.0, 0
        for X, y, *_ in loader:
            X, y = X.to(device), y.to(device)
            params = _get_fp_params(model)  # sync container after optimizer.step
            optimizer.zero_grad()
            grads = _per_sample_grad_dp(params, X, y, model)
            # .detach(): the fp64 copies must not keep the vmap graph alive
            norms = torch.sqrt(sum(
                g.detach().double().pow(2).sum(dim=tuple(range(1, g.dim())))
                for g in grads.values()))
            scale = (clipping_norm / norms.clamp_min(1e-12)).clamp_max(1.0).float()
            b = X.shape[0]
            # 未裁剪样本（norm ≤ C，scale==1）占比——与 Flower 几何更新公式方向一致
            n_unclipped += int((scale >= 1.0).sum().item())
            n_samples += b
            for name, p in model.named_parameters():
                g = grads[name]
                p.grad = (g * scale.view(b, *([1] * (g.dim() - 1)))).sum(0) / b
                p.grad += torch.randn_like(p.grad) * (
                    noise_multiplier * clipping_norm / b)
            optimizer.step()
            with torch.no_grad():
                out = model(X)
                epoch_loss += (out - y).abs().sum().item()
                epoch_samples += b
        loss = epoch_loss / max(epoch_samples, 1)
    # Return CUDA segments to the allocator so the next round starts with a
    # clean slate (the container itself is kept for the process lifetime).
    del optimizer
    import gc; gc.collect()
    torch.cuda.empty_cache()
    fraction = (n_unclipped / n_samples) if n_samples > 0 else 0.0
    return loss, float(fraction)


def train_client(tensors: list, keys: list[str], model, train_ds, n_train: int,
                 cfg: dict, dp: dict | None) -> dict:
    """One round of local training from ``tensors`` (model mutated in place)."""
    state = tensors_to_state_dict(tensors, keys)
    model.load_state_dict(state)
    model.to(cfg["device"])
    sigma: float | None = None
    loss: float = float("nan")
    clip_fraction: float | None = None
    if dp is not None:
        if dp["mode"] == "per_client":
            sigma, _ = sigma_for_epsilon(
                n_train, cfg["batch_size"], cfg["local_epochs"],
                cfg["rounds"], dp["delta"], dp["target_epsilon"])
        else:
            sigma = float(dp["sigma"])
        eps_sigma = sigma   # 会计锚点：预支修正前的目标 σ
        if dp.get("adaptive_clip"):
            sigma = adaptive_sigma_train(sigma, dp["clip_count_noise"])
        loss, clip_fraction = _train_dp(
            model, train_ds, cfg["lr"], cfg["batch_size"],
            cfg["local_epochs"], cfg["device"], sigma,
            dp["clipping_norm"])
    else:
        loss = _train_plain(model, train_ds, cfg["lr"], cfg["batch_size"],
                            cfg["local_epochs"], cfg["device"])
    eps = None
    if dp is not None:
        eps = dp_epsilon(n_train, cfg["batch_size"], cfg["local_epochs"],
                         cfg["round"], eps_sigma, dp["delta"])
    return {"tensors": state_dict_to_tensors(model.state_dict()),
            "n_train": n_train, "eps": eps, "sigma": sigma,
            "loss": loss, "clip_fraction": clip_fraction}


class CidEchoClient(NumPyClient):
    """Wraps a FedClient to echo its semantic client id in fit metrics.

    In flwr 1.x the server-side proxy cid is an opaque (random) node id
    in both simulation and the App line, so the strategy cannot map a
    result back to its participant without the client's cooperation.
    """

    def __init__(self, inner: FedClient, cid: str) -> None:
        self._inner = inner
        self._cid = cid

    def fit(self, parameters, config):
        tensors, n_train, metrics = self._inner.fit(parameters, config)
        return tensors, n_train, {**metrics, "cid": self._cid}

    def evaluate(self, parameters, config):
        return self._inner.evaluate(parameters, config)

    def get_parameters(self, config):
        return self._inner.get_parameters(config)


class FedClient(NumPyClient):
    """flwr client wrapping one participant's data + local DP training."""

    def __init__(self, cache: dict, state_keys: list[str], cfg: dict) -> None:
        self.cache = cache
        self.state_keys = state_keys
        self.cfg = cfg
        self.budget_history: list[dict] = []
        self.budget_path: Path | None = cfg.get("budget_path")
        self._sigma_cache: float | None = None

    def get_parameters(self, config):
        return state_dict_to_tensors(build_tcn(TCNConfig()).state_dict())

    def fit(self, parameters, config):
        dp = None
        if config.get("dp_mode") not in (None, "none", ""):
            if config["dp_mode"] == "per_client":
                # σ depends only on (n, batch, epochs, rounds, δ, ε) — all
                # fixed for a run — so derive once and reuse across rounds
                # instead of recomputing the PLD search every round.
                if self._sigma_cache is None:
                    sigma, _ = sigma_for_epsilon(
                        self.cache["n_train"], self.cfg["batch_size"],
                        self.cfg["local_epochs"], int(config["rounds"]),
                        float(config["dp_delta"]),
                        float(config["dp_target_epsilon"]))
                    self._sigma_cache = float(sigma)
                # Mathematically equivalent to per_client mode: train_client's
                # ε/loss math does not depend on the mode field (the per_client
                # branch is kept for direct callers).
                dp = {"mode": "uniform",
                      "clipping_norm": float(config["dp_clip"]),
                      "delta": float(config["dp_delta"]),
                      "sigma": self._sigma_cache,
                      "target_epsilon": float(config["dp_target_epsilon"])}
            else:
                dp = {"mode": config["dp_mode"],
                      "clipping_norm": float(config["dp_clip"]),
                      "delta": float(config["dp_delta"]),
                      "sigma": float(config.get("dp_sigma") or 0.0),
                      "target_epsilon": float(config.get("dp_target_epsilon") or 0.0)}
        round_cfg = {**self.cfg,
                     "round": int(config["server_round"]),
                     "rounds": int(config["rounds"])}
        # Fresh model per round: a module used inside a torch.func
        # vmap/grad transform can no longer pass Module._apply (model.to()),
        # which train_client calls — reusing one model across rounds would
        # crash DP round 2+ in long-lived clients (the App line).
        model = build_tcn(TCNConfig())
        result = train_client(parameters, self.state_keys, model,
                              self.cache["train_ds"], self.cache["n_train"],
                              round_cfg, dp)
        if result["eps"] is not None:
            self.budget_history.append(
                {"round": round_cfg["round"], "eps": result["eps"],
                 "sigma": result["sigma"]})
            self._write_budget()
        metrics = {"loss": float(result["loss"])}
        if result["eps"] is not None:
            metrics["eps"] = float(result["eps"])
        return (result["tensors"], result["n_train"], metrics)

    def evaluate(self, parameters, config):
        if self.cfg.get("deliver_model") and self.budget_path is not None:
            state = tensors_to_state_dict(parameters, self.state_keys)
            torch.save(state, self.budget_path.parent / "final_model.pt")
        return 0.0, self.cache["n_train"], {}

    def _write_budget(self) -> None:
        if self.budget_path is None:
            return
        self.budget_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.budget_path, "w") as f:
            json.dump({"rounds": self.budget_history}, f, indent=2)
