"""FedAvg aggregation strategy + audit log + per-round checkpoints."""
from __future__ import annotations

import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import torch
from flwr.client import NumPyClient
from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

from fl_code.fed_core.client_core import FedClient
from fl_code.fed_core.params import state_dict_to_tensors, tensors_to_state_dict
from fl_code.models import TCNConfig, build_tcn


def initial_parameters(state_keys: list[str]):
    model = build_tcn(TCNConfig())
    return ndarrays_to_parameters(state_dict_to_tensors(model.state_dict()))


class _CidEchoClient(NumPyClient):
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


class AuditFedAvg(FedAvg):
    """FedAvg + per-round audit rows + checkpoint saving (post-processing)."""

    def __init__(self, *, task: dict, state_keys: list[str],
                 on_round_done=None, **fedavg_kwargs) -> None:
        super().__init__(**fedavg_kwargs)
        self.task = task
        self.state_keys = state_keys
        self.on_round_done = on_round_done
        self.audit_rows: list[dict] = []
        Path(task["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    def configure_fit(self, server_round, parameters, client_manager):
        cfg = {**self.task["cfg"],
               "server_round": server_round,
               "rounds": self.task["rounds"]}
        # flwr ConfigRecord rejects None values; FedClient reads missing
        # keys via .get(...) or 0.0, so dropping None keys is lossless.
        cfg = {k: v for k, v in cfg.items() if v is not None}
        return [(client, FitIns(parameters, cfg))
                for client in client_manager.all().values()]

    def aggregate_fit(self, server_round, results, failures):
        params, _ = super().aggregate_fit(server_round, results, failures)
        joined = []
        client_losses = {}
        for proxy, res in results:
            cid = str(res.metrics.get("cid") or proxy.cid)
            joined.append(cid)
            client_losses[cid] = float(res.metrics.get("loss", 0.0))
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
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.audit_rows.append(row)
        self._write_audit()
        if params is not None:
            state = OrderedDict(tensors_to_state_dict(
                parameters_to_ndarrays(params), self.state_keys))
            torch.save(state, Path(self.task["checkpoint_dir"])
                       / f"round_{server_round:03d}.pt")
        if self.on_round_done is not None:
            self.on_round_done(row)
        return params, {}

    def _write_audit(self) -> None:
        path = Path(self.task["audit_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"task": {"name": self.task["name"],
                                "rounds": self.task["rounds"],
                                "round_timeout": self.task.get("round_timeout"),
                                "started_at": self.task.get("started_at")},
                       "rounds": self.audit_rows}, f, indent=2,
                      ensure_ascii=False)


def build_strategy(task: dict, state_keys: list[str], on_round_done=None):
    return AuditFedAvg(
        task=task, state_keys=state_keys, on_round_done=on_round_done,
        fraction_fit=1.0,
        fraction_evaluate=1.0 if task.get("deliver_model") else 0.0,
        min_fit_clients=1, min_evaluate_clients=1, min_available_clients=1,
        accept_failures=True,
        initial_parameters=initial_parameters(state_keys),
        fit_metrics_aggregation_fn=(
            lambda metrics: {"loss": float(sum(m[1].get("loss", 0.0) * m[0]
                                               for m in metrics)
                                           / max(sum(m[0] for m in metrics), 1))}),
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
        return _CidEchoClient(FedClient(caches[cid], state_keys,
                                        {**cfg, "budget_path": None}),
                              cid).to_client()
    return client_fn
