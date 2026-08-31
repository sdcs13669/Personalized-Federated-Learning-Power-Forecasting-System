"""FedAvg aggregation strategy + audit log + per-round checkpoints."""
from __future__ import annotations

import json
import logging
import math
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import torch
from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

from fl_code.fed_core.client_core import CidEchoClient, FedClient
from fl_code.fed_core.params import state_dict_to_tensors, tensors_to_state_dict
from fl_code.models import TCNConfig, build_tcn


def initial_parameters(state_keys: list[str]):
    model = build_tcn(TCNConfig())
    return ndarrays_to_parameters(state_dict_to_tensors(model.state_dict()))


class AuditFedAvg(FedAvg):
    """FedAvg + per-round audit rows + checkpoint saving (post-processing)."""

    def __init__(self, *, task: dict, state_keys: list[str],
                 on_round_done=None, **fedavg_kwargs) -> None:
        super().__init__(**fedavg_kwargs)
        self.task = task
        self.state_keys = state_keys
        self.on_round_done = on_round_done
        self.audit_rows: list[dict] = []
        self._last_parameters = None
        ckpt_dir = task.get("checkpoint_dir")
        if ckpt_dir:
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        # Adaptive clipping (Andrew et al. 2021) state.  Off unless the task
        # opts in via cfg["dp_adaptive_clip"]; when off, configure_fit and
        # aggregate_fit behave exactly as before.
        cfg = task.get("cfg", {})
        self.adaptive_clip = bool(cfg.get("dp_adaptive_clip", False))
        self._clip_norm = float(cfg.get("dp_clip", 1.0) or 1.0)
        self.clip_lr = float(cfg.get("dp_clip_lr", 0.2))
        self.clip_target = float(cfg.get("dp_clip_target_quantile", 0.5))
        self.clip_count_noise = float(cfg.get("dp_clip_count_noise", 0.5))

    def configure_fit(self, server_round, parameters, client_manager):
        cfg = {**self.task["cfg"],
               "server_round": server_round,
               "rounds": self.task["rounds"]}
        # flwr ConfigRecord rejects None values; FedClient reads missing
        # keys via .get(...) or 0.0, so dropping None keys is lossless.
        cfg = {k: v for k, v in cfg.items() if v is not None}
        if self.adaptive_clip:
            cfg["dpfedavg_clip_norm"] = self._clip_norm
            cfg["dpfedavg_clip_count_noise"] = self.clip_count_noise
        return [(client, FitIns(parameters, cfg))
                for client in client_manager.all().values()]

    def aggregate_fit(self, server_round, results, failures):
        params, _ = super().aggregate_fit(server_round, results, failures)
        arrived = set()
        client_losses = {}
        client_epsilons = {}
        for proxy, res in results:
            cid = res.metrics.get("cid")
            if cid is None:
                # No silent random-node-id fallback: a result without a
                # semantic cid echo cannot be attributed to a participant,
                # so it must not enter the audit (joined/loss).
                logging.getLogger("flwr").warning(
                    "proxy %s did not echo a semantic client cid; "
                    "result excluded from audit round %d",
                    proxy.cid, server_round)
                continue
            cid = str(cid)
            arrived.add(cid)
            client_losses[cid] = float(res.metrics.get("loss", 0.0))
            eps = res.metrics.get("eps")
            if eps is not None:
                client_epsilons[cid] = round(float(eps), 6)
        # Deterministic order (expected_clients) so audits diff cleanly
        # across runs; unexpected cids are appended at the tail.
        joined = [cid for cid in self.task["expected_clients"]
                  if cid in arrived]
        joined += [cid for cid in arrived if cid not in joined]
        dropped = [cid for cid in self.task["expected_clients"]
                   if cid not in joined]
        row = {
            "round": server_round,
            "expected": list(self.task["expected_clients"]),
            "joined": joined,
            "dropped": dropped,
            "loss": round(float(sum(client_losses.values())
                                / max(len(client_losses), 1)), 6),
            "client_losses": client_losses,
            "client_epsilons": client_epsilons,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
        if self.adaptive_clip:
            # Pre-update bound: the bound that was actually used this round.
            row["clip_norm"] = round(self._clip_norm, 6)
        self.audit_rows.append(row)
        self._write_audit()
        if self.adaptive_clip:
            self._update_clip_norm(results)
        if params is not None:
            self._last_parameters = parameters_to_ndarrays(params)
            ckpt_dir = self.task.get("checkpoint_dir")
            if ckpt_dir:
                state = OrderedDict(tensors_to_state_dict(
                    self._last_parameters, self.state_keys))
                torch.save(state, Path(ckpt_dir)
                           / f"round_{server_round:03d}.pt")
        if self.on_round_done is not None:
            self.on_round_done(row)
        return params, {}

    def _update_clip_norm(self, results) -> None:
        """Geometric update (Andrew et al. 2021): C ← C·exp(−η(f̃−τ)).

        f̃ = equal-weight average of the noised per-client UNCLIPPED-sample
        fractions (semantics matches Flower's formula direction: f̃>τ means
        most samples are within the bound → C shrinks).  Clients that did
        not report are skipped; no update when nobody reported.
        """
        fractions = []
        for proxy, res in results:
            f = res.metrics.get("dpfedavg_clip_fraction")
            if f is None:
                continue
            f = float(f)
            if not math.isfinite(f):
                # Poisoned/non-finite fraction: NaN propagates through the
                # mean and the min/max clamp, corrupting _clip_norm for the
                # rest of the run — skip it.
                logging.getLogger("flwr").warning(
                    "proxy %s reported non-finite dpfedavg_clip_fraction=%r; "
                    "skipped in clip-norm update", proxy.cid, f)
                continue
            fractions.append(f)
        if not fractions:
            return
        f_avg = sum(fractions) / len(fractions)
        new_norm = self._clip_norm * math.exp(
            -self.clip_lr * (f_avg - self.clip_target))
        self._clip_norm = float(min(max(new_norm, 1e-4), 1e4))

    def _write_audit(self) -> None:
        audit_path = self.task.get("audit_path")
        if not audit_path:
            return
        path = Path(audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"task": {"name": self.task["name"],
                                "rounds": self.task["rounds"],
                                "round_timeout": self.task.get("round_timeout"),
                                "started_at": self.task.get("started_at")},
                       "rounds": self.audit_rows}, f, indent=2,
                      ensure_ascii=False)


def build_strategy(task: dict, state_keys: list[str], on_round_done=None):
    # 联邦学习必须等齐所有参与方再开跑：min_* 按 expected_clients 数量设，
    # 否则一个客户端先连上就开跑、后加入的从中间轮开始，导致演示乱序。
    expected = task.get("expected_clients", [])
    n_clients = max(len(expected), 1) if expected else 1
    return AuditFedAvg(
        task=task, state_keys=state_keys, on_round_done=on_round_done,
        fraction_fit=1.0,
        fraction_evaluate=1.0 if task.get("deliver_model") else 0.0,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        accept_failures=True,
        initial_parameters=initial_parameters(state_keys),
    )


def build_server_app(task: dict, state_keys: list[str]) -> ServerApp:
    return ServerApp(
        strategy=build_strategy(task, state_keys),
        config=ServerConfig(num_rounds=task["rounds"],
                            round_timeout=task.get("round_timeout")),
    )


def make_client_fn(caches: dict, client_ids: list[str],
                   state_keys: list[str], cfg: dict):
    def client_fn(context):
        i = int(context.node_config["partition-id"])
        cid = client_ids[i]
        return CidEchoClient(FedClient(caches[cid], state_keys,
                                       {**cfg, "budget_path": None}),
                             cid).to_client()
    return client_fn
