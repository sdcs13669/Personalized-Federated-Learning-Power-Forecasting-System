"""flwr server worker：独立子进程运行 flwr server。

解决 flwr 1.30 `start_server` 在后台线程无法监听 gRPC 的问题——
独立进程主线程里 start_server 能正常监听。

用法: python server/fl_server_worker.py <task.pkl> <model_out.pkl>
  task.pkl    : pickle dict {id, name, rounds, round_timeout, grpc_port, cfg, participants}
  model_out.pkl: 训练完成后写入 {keys, tensors}（pickle），供 get_final_model 读取
"""
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("fl_worker")

ROOT = Path(__file__).resolve().parent.parent
# 子进程运行时 cwd 不在 sys.path，需手动把项目根加进来才能 import fl_code/server
sys.path.insert(0, str(ROOT))


def main() -> None:
    task_pkl = Path(sys.argv[1])
    model_out = Path(sys.argv[2])

    with open(task_pkl, "rb") as f:
        task_dict = pickle.load(f)

    task_id = task_dict["id"]
    participants = task_dict["participants"]

    from flwr.server import ServerConfig, start_server

    from fl_code.fed_core.server_core import build_strategy
    from fl_code.models import TCNConfig, build_tcn

    from server.database import SessionLocal, init_db
    from server.models import AuditRound, Task

    model = build_tcn(TCNConfig())
    state_keys = list(model.state_dict().keys())
    expected_clients = [p["client_id"] for p in participants]

    init_db()
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = "training"
            task.started_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()

    def on_round_done(row: dict) -> None:
        db = SessionLocal()
        try:
            audit = AuditRound(
                task_id=task_id,
                round=row["round"],
                expected=json.dumps(row["expected"]),
                joined=json.dumps(row["joined"]),
                dropped=json.dumps(row["dropped"]),
                loss=row.get("loss"),
                client_losses=json.dumps(row.get("client_losses", {})),
                client_epsilons=json.dumps(row.get("client_epsilons", {})),
                clip_norm=row.get("clip_norm"),
                finished_at=datetime.utcnow(),
            )
            db.add(audit)
            db.commit()
        except Exception as e:
            logger.error("audit write failed: %s", e)
        finally:
            db.close()

    flwr_task = {
        "name": task_dict["name"],
        "rounds": task_dict["rounds"],
        "round_timeout": task_dict.get("round_timeout"),
        "checkpoint_dir": None,
        "audit_path": None,
        "expected_clients": expected_clients,
        "deliver_model": True,
        "started_at": str(datetime.utcnow()),
        "cfg": task_dict.get("cfg", {}),
    }
    strategy = build_strategy(flwr_task, state_keys, on_round_done=on_round_done)

    grpc_port = task_dict.get("grpc_port", 8089)
    server_error = None
    try:
        start_server(
            server_address=f"0.0.0.0:{grpc_port}",
            strategy=strategy,
            config=ServerConfig(
                num_rounds=task_dict["rounds"],
                round_timeout=task_dict.get("round_timeout"),
            ),
        )
    except Exception as e:
        logger.error("flwr server error: %s", e)
        server_error = e

    # 保存最终模型，供 get_final_model 读取
    try:
        if server_error is None and strategy._last_parameters is not None:
            lp = strategy._last_parameters
            # AuditFedAvg 已把 _last_parameters 转成 ndarray 列表
            tensors = lp if isinstance(lp, list) else parameters_to_ndarrays(lp)
            with open(model_out, "wb") as f:
                pickle.dump({"keys": state_keys, "tensors": tensors}, f)
    except Exception as e:
        logger.error("save model failed: %s", e)

    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = "failed" if server_error else "completed"
            task.finished_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()

    logger.info("flwr worker finished for task %d: %s",
                task_id, "failed" if server_error else "completed")


if __name__ == "__main__":
    main()
