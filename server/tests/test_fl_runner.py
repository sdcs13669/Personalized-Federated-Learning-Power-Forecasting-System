"""Test fl_runner task lifecycle (unit level, no real flwr)."""
from server.fl_runner import ActiveTask, _active_tasks, get_final_model


def test_get_final_model_returns_none_when_no_task():
    assert get_final_model(999) is None


def test_get_final_model_returns_bytes():
    import numpy as np
    task = ActiveTask(task_id=42, thread=None, stop_event=None)
    task.final_tensors = [np.zeros((2, 2), dtype=np.float32)]
    task.state_keys = ["weight"]
    _active_tasks[42] = task
    result = get_final_model(42)
    assert result is not None
    import pickle
    data = pickle.loads(result)
    assert data["keys"] == ["weight"]
    assert len(data["tensors"]) == 1
    del _active_tasks[42]
