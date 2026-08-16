"""End-to-end: 2 clients, 1 round, via run_simulation (ray backend)."""
import json

import torch
from flwr.client import ClientApp
from flwr.simulation import run_simulation

from fl_code.fed_core.client_core import FedClient
from fl_code.fed_core.data import load_client_cache
from fl_code.fed_core.server_core import build_server_app, make_client_fn
from fl_code.models import TCNConfig, build_tcn


def test_two_client_simulation(tmp_path):
    client_ids = ["steel_ind_0", "tetouan_city_0"]
    caches = {cid: load_client_cache(cid, stride=6, max_seqs=1)
              for cid in client_ids}
    keys = list(build_tcn(TCNConfig()).state_dict().keys())
    task = {
        "name": "smoke", "rounds": 1, "round_timeout": None,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "audit_path": str(tmp_path / "audit_log.json"),
        "expected_clients": client_ids, "deliver_model": False,
        "started_at": None,
        "cfg": {"lr": 0.001, "batch_size": 16, "local_epochs": 1,
                "dp_mode": "none", "dp_clip": 1.0, "dp_delta": 1e-5,
                "dp_sigma": None, "dp_target_epsilon": None},
    }
    server_app = build_server_app(task, keys)
    client_app = ClientApp(client_fn=make_client_fn(
        caches, client_ids, keys,
        {"lr": 0.001, "batch_size": 16, "local_epochs": 1,
         "device": "cpu"}))
    run_simulation(server_app=server_app, client_app=client_app,
                   num_supernodes=2,
                   backend_config={"client_resources": {"num_cpus": 1,
                                                        "num_gpus": 0}})
    audit = json.loads((tmp_path / "audit_log.json").read_text())
    assert len(audit["rounds"]) == 1
    assert set(audit["rounds"][0]["joined"]) == set(client_ids)
    assert audit["rounds"][0]["dropped"] == []
    ckpt = tmp_path / "checkpoints" / "round_001.pt"
    assert ckpt.exists()
    state = torch.load(ckpt, weights_only=True)
    assert list(state.keys()) == keys
