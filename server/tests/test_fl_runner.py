"""Test fl_runner task lifecycle (unit level, no real flwr).

注意：flwr 早前从「后台线程」改为「独立子进程 worker」，因此
  · get_final_model 不再读内存里的 final_tensors，而是读 worker 落盘的 pkl；
  · ActiveTask 不再有 thread/stop_event，只有 proc。
本文件按当前实现（子进程模型）重写。
"""
import pickle

import numpy as np

import server.fl_runner as fl_runner
from server.fl_runner import (ActiveTask, _active_tasks, get_final_model,
                              get_task_status)


class _FakeProc:
    """subprocess.Popen 的最小替身：只需 poll() 返回码。"""

    def __init__(self, returncode):
        self.returncode = returncode

    def poll(self):
        return self.returncode


def test_get_final_model_returns_none_when_no_task(tmp_path, monkeypatch):
    monkeypatch.setattr(fl_runner, "MODEL_DIR", tmp_path)
    assert get_final_model(999) is None


def test_get_final_model_returns_bytes(tmp_path, monkeypatch):
    """worker 把模型写进 MODEL_DIR/fl_model_<id>.pkl，主进程应读回同样的字节。"""
    monkeypatch.setattr(fl_runner, "MODEL_DIR", tmp_path)
    payload = {"keys": ["weight"],
               "tensors": [np.zeros((2, 2), dtype=np.float32)]}
    (tmp_path / "fl_model_42.pkl").write_bytes(pickle.dumps(payload))

    result = get_final_model(42)
    assert result is not None
    data = pickle.loads(result)
    assert data["keys"] == ["weight"]
    assert len(data["tensors"]) == 1


def test_model_path_follows_model_dir(tmp_path, monkeypatch):
    """_model_path 应跟随 MODEL_DIR（Docker 是 /app/data，本机是仓库 data 目录）。"""
    monkeypatch.setattr(fl_runner, "MODEL_DIR", tmp_path)
    assert fl_runner._model_path(7) == tmp_path / "fl_model_7.pkl"


def test_get_task_status_none_when_not_active():
    assert get_task_status(4242) is None


def test_get_task_status_training_completed_failed():
    task_id = 7
    _active_tasks[task_id] = ActiveTask(task_id=task_id,
                                        proc=_FakeProc(returncode=None))
    try:
        assert get_task_status(task_id) == "training"

        _active_tasks[task_id].proc = _FakeProc(returncode=0)
        assert get_task_status(task_id) == "completed"

        _active_tasks[task_id].proc = _FakeProc(returncode=1)
        assert get_task_status(task_id) == "failed"
    finally:
        del _active_tasks[task_id]


def test_get_task_status_none_when_proc_missing():
    task_id = 8
    _active_tasks[task_id] = ActiveTask(task_id=task_id, proc=None)
    try:
        assert get_task_status(task_id) is None
    finally:
        del _active_tasks[task_id]
