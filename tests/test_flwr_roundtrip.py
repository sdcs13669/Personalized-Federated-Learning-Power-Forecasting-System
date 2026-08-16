"""Minimal flwr client-server roundtrip: proves the transport works."""
import threading
import time

import flwr.compat.server.app as compat_server_app
from flwr.client import Client, NumPyClient, start_client
from flwr.common import Code, FitIns, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvg


class EchoClient(NumPyClient):
    def fit(self, parameters, config):
        assert config["server_round"] == 1
        assert config["payload"] == "hello"
        return parameters, 2, {"loss": 0.5}

    def evaluate(self, parameters, config):
        return 0.0, 2, {}


def test_roundtrip():
    port = 8097

    # flwr 1.30's start_server() unconditionally registers SIGINT/SIGTERM
    # handlers for interactive Ctrl-C shutdown, which raises
    # "ValueError: signal only works in main thread of the main interpreter"
    # when the server runs in a worker thread (as it does in this in-process
    # test). Disable that registration: it is irrelevant for a 1-round test.
    compat_server_app.register_signal_handlers = lambda *args, **kwargs: None

    server_thread = threading.Thread(
        target=start_server,
        kwargs={
            "server_address": f"127.0.0.1:{port}",
            "config": ServerConfig(num_rounds=1),
            "strategy": FedAvg(
                min_fit_clients=1, min_evaluate_clients=1,
                min_available_clients=1,
                initial_parameters=ndarrays_to_parameters([], ),
                on_fit_config_fn=lambda r: {"server_round": r, "payload": "hello"},
            ),
        },
        daemon=True,
    )
    server_thread.start()
    time.sleep(3.0)

    client: Client = EchoClient().to_client()
    start_client(server_address=f"127.0.0.1:{port}", client=client)

    server_thread.join(timeout=10)
    assert not server_thread.is_alive(), "server did not finish"
