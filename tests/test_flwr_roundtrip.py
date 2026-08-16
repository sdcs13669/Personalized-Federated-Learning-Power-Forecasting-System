"""Minimal flwr client-server roundtrip: proves the transport works."""
import socket
import threading
import time

import pytest
import flwr.compat.server.app as compat_server_app
from flwr.client import Client, NumPyClient, start_client
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvg


class EchoClient(NumPyClient):
    def fit(self, parameters, config):
        assert config["server_round"] == 1
        assert config["payload"] == "hello"
        return parameters, 2, {"loss": 0.5}

    def evaluate(self, parameters, config):
        return 0.0, 2, {}


def _wait_for_server(port: int, timeout: float = 15.0) -> None:
    """Poll the server address until it accepts connections (ready)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.2)
    pytest.fail("server not ready")


def test_roundtrip(monkeypatch):
    port = 8097

    # flwr 1.30's start_server() unconditionally registers SIGINT/SIGTERM
    # handlers for interactive Ctrl-C shutdown, which raises
    # "ValueError: signal only works in main thread of the main interpreter"
    # when the server runs in a worker thread (as it does in this in-process
    # test). Disable that registration: it is irrelevant for a 1-round test.
    # monkeypatch fixture undoes this after the test.
    monkeypatch.setattr(compat_server_app, "register_signal_handlers",
                        lambda *args, **kwargs: None)

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
    _wait_for_server(port)

    client: Client = EchoClient().to_client()
    client_thread = threading.Thread(
        target=start_client,
        kwargs={"server_address": f"127.0.0.1:{port}", "client": client},
        daemon=True,
    )
    client_thread.start()
    client_thread.join(timeout=60)
    assert not client_thread.is_alive(), "client did not finish"

    server_thread.join(timeout=10)
    assert not server_thread.is_alive(), "server did not finish"
