"""client_agent：本地 HTTP 代理（静态前端 + REST 转发 + 本地控制端点）。

启动: python app/agent.py
浏览器打开 http://localhost:9001
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 保证 `python app/agent.py` 运行时能 import app.*
DEFAULT_WEB = ROOT / "web"
_APP_DIR = Path(__file__).resolve().parent
# 同一仓库可跑多个客户端：FL_AGENT_CONFIG 指定各自的配置文件
CONFIG_PATH = Path(os.environ.get("FL_AGENT_CONFIG",
                                  str(_APP_DIR / "agent_config.json")))
# 数据目录：默认 app/data，可用 FL_DATA_DIR 或配置里的 data_dir 覆盖
DATA_DIR = Path(os.environ.get("FL_DATA_DIR", str(_APP_DIR / "data")))

_default_config = {
    "server_url": "http://127.0.0.1:8000",
    "username": "",
    "password": "",
    "client_id": "",
    "local_port": 9001,
    "data_dir": "",   # 空 = 默认 app/data；多客户端时指向各自的数据目录
}

# Task 1 演示数据集兜底清单（GitHub raw URL）。server 端 /api/datasets（Task 7）
# 就绪前先用这份本地清单让 /local/collect 可用。
DATASETS_FALLBACK = [
    {"id": "steel_ind_0", "client_id": "steel_ind_0",
     "url": "https://raw.githubusercontent.com/sdcs13669/Personalized-Federated-Learning-Power-Forecasting-System/main/data/app_datasets/steel_ind_0.zip"},
    {"id": "tetouan_0", "client_id": "tetouan_city_0",
     "url": "https://raw.githubusercontent.com/sdcs13669/Personalized-Federated-Learning-Power-Forecasting-System/main/data/app_datasets/tetouan_0.zip"},
    {"id": "tetouan_1", "client_id": "tetouan_city_1",
     "url": "https://raw.githubusercontent.com/sdcs13669/Personalized-Federated-Learning-Power-Forecasting-System/main/data/app_datasets/tetouan_1.zip"},
    {"id": "tetouan_2", "client_id": "tetouan_city_2",
     "url": "https://raw.githubusercontent.com/sdcs13669/Personalized-Federated-Learning-Power-Forecasting-System/main/data/app_datasets/tetouan_2.zip"},
]


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        return {**_default_config, **cfg}
    return dict(_default_config)


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def _forward(method: str, url: str, headers: dict, body: bytes | None) -> object:
    """转发到远程 server；返回带 status_code/headers/json() 的响应对象。

    注意：urllib 对非 2xx 会抛 HTTPError，这里捕获后也封装成响应对象，
    这样调用方（local_login / forward）能按状态码正确处理而不是误判为断网。
    """
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    # 强制直连不走系统代理：server 走 Tailscale/内网地址，Clash 系统代理会劫持导致连不上
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        resp = opener.open(req, timeout=30)
        raw = resp.read()
        status = resp.status
        resp_headers = dict(resp.headers)
        content_type = resp.headers.get("content-type", "")
    except urllib.error.HTTPError as e:
        raw = e.read()
        status = e.code
        resp_headers = dict(e.headers)
        content_type = e.headers.get("content-type", "")

    class _Resp:
        status_code = status
        headers = resp_headers

        def json(self):
            return json.loads(raw)

        @property
        def content(self):
            return raw

    return _Resp()


def _fetch_datasets(server_url: str, token: str | None) -> list[dict]:
    """从远程 server 拉数据集清单；失败返回空列表（调用方有本地兜底）。"""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    try:
        resp = _forward("GET", server_url + "/api/datasets", headers, None)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


class LoginBody(BaseModel):
    username: str
    password: str


class CollectBody(BaseModel):
    dataset_id: str


class TrainBody(BaseModel):
    grpc_addr: str


class RcBody(BaseModel):
    task_id: int


def create_app(web_dir: str | None = None,
               server_url: str | None = None,
               config_path: Path | None = None) -> FastAPI:
    global DATA_DIR
    web_dir = web_dir or str(DEFAULT_WEB)
    cfg = load_config()
    if cfg.get("data_dir"):
        DATA_DIR = Path(cfg["data_dir"]).resolve()
    os.environ["FL_DATA_DIR"] = str(DATA_DIR)   # trainer 也读同一数据目录
    if server_url:
        cfg["server_url"] = server_url
    if config_path:
        # 测试注入配置路径
        cfg["_config_path"] = str(config_path)

    app = FastAPI(title="FL Client Agent")
    token = {"value": None}

    @app.get("/local/status")
    def local_status():
        data_dir = DATA_DIR
        collected = (data_dir / "dataset_id.txt").exists()
        dataset_id = None
        if collected:
            dataset_id = (data_dir / "dataset_id.txt").read_text(
                encoding="utf-8").strip()
        return {
            "online": True,
            "server_url": cfg["server_url"],
            "username": cfg.get("username", ""),
            "client_id": cfg.get("client_id", ""),
            "data_collected": collected,
            "dataset_id": dataset_id,
        }

    @app.post("/local/login")
    def local_login(body: LoginBody):
        # 用远程 server 校验并存储 token（只存内存 + 本地文件）
        try:
            resp = _forward(
                "POST", cfg["server_url"] + "/api/auth/login",
                {"Content-Type": "application/json"},
                json.dumps({"username": body.username,
                            "password": body.password}).encode(),
            )
        except urllib.error.URLError:
            return JSONResponse(status_code=502,
                                content={"detail": "无法连接远程服务器"})
        if resp.status_code != 200:
            return JSONResponse(status_code=401,
                                content={"detail": "登录失败"})
        token["value"] = resp.json()["access_token"]
        cfg["username"] = body.username
        cfg["password"] = body.password
        save_config(cfg)
        return {"ok": True}

    @app.get("/local/token")
    def local_token():
        return {"token": token["value"]}

    @app.post("/local/collect")
    def local_collect(body: CollectBody):
        try:
            from app.collector import collect_dataset
            datasets = _fetch_datasets(cfg["server_url"], token["value"]) \
                or DATASETS_FALLBACK
            ds = next((d for d in datasets if d["id"] == body.dataset_id), None)
            if ds is None:
                return JSONResponse(status_code=404,
                                    content={"detail": "未知数据集"})
            if ds.get("client_id") and cfg.get("client_id") and \
                    ds["client_id"] != cfg["client_id"]:
                return JSONResponse(
                    status_code=400,
                    content={"detail": f"该数据源属于 {ds['client_id']}，"
                                      f"与你的角色 {cfg.get('client_id')} 不匹配"})
            info = collect_dataset(body.dataset_id, ds["url"], DATA_DIR,
                                   cfg.get("client_id", ""))
            return info
        except Exception as e:
            return JSONResponse(status_code=500,
                                content={"detail": f"采集失败: {e}"})

    @app.post("/local/start")
    def local_start(body: TrainBody):
        from app.trainer import start_training
        try:
            msg = start_training(body.grpc_addr, cfg.get("client_id", ""), {
                "batch_size": 64, "local_epochs": 1, "lr": 0.001,
                "device": "cpu",
            })
            return {"ok": True, "message": msg}
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": str(e)})

    @app.get("/local/train-status")
    def local_train_status():
        from app.trainer import get_train_status
        return get_train_status()

    @app.post("/local/rc")
    def local_rc(body: RcBody):
        from app.rc_runner import (download_model_bytes, parse_model_bytes,
                                   run_rc, upload_rc_result)
        t = token["value"]
        if not t:
            return JSONResponse(status_code=401,
                                content={"detail": "未登录"})
        try:
            raw = download_model_bytes(cfg["server_url"], t, body.task_id)
            keys, tensors = parse_model_bytes(raw)
        except Exception as e:
            return JSONResponse(status_code=500,
                                content={"detail": f"下载模型失败: {e}"})
        work = DATA_DIR / "rc_work"
        work.mkdir(parents=True, exist_ok=True)
        model_pt = work / "global_model.pt"
        import torch
        from fl_code.fed_core.params import tensors_to_state_dict
        from fl_code.models import TCNConfig, build_tcn
        state = tensors_to_state_dict(tensors, keys)
        torch.save(state, model_pt)
        out = DATA_DIR / "rc_out"
        cmd = [sys.executable, "-m", "fl_code.train_personalized",
               "--global-model", str(model_pt),
               "--output-dir", str(out),
               "--data-dir", str(DATA_DIR),
               "--clients", cfg.get("client_id", "")]
        try:
            run_rc(cmd, ROOT)
        except Exception as e:
            return JSONResponse(status_code=500,
                                content={"detail": f"RC 训练失败: {e}"})
        import json
        res_json = out / "personalized_results.json"
        if res_json.exists():
            results = json.loads(res_json.read_text(encoding="utf-8"))
        else:
            results = {}
        wg = results.get("wape_global")
        wr = results.get("wape_rc")
        png = next((p for p in out.rglob("*.png")), None)
        ok = upload_rc_result(cfg["server_url"], t, body.task_id,
                              cfg.get("client_id", ""), wg or 0, wr or 0,
                              str(png) if png else None)
        return {"ok": ok, "wape_global": wg, "wape_rc": wr,
                "png": str(png) if png else None}

    # ===== need.md 阶段2/3：本地二阶段 + 预测展示 =====
    @app.get("/local/stage-status")
    def local_stage_status(task_id: int):
        from app.client_stage import get_status
        try:
            return get_status(task_id)
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": str(e)})

    @app.post("/local/stage2")
    def local_stage2(body: RcBody):
        from app.client_stage import run_stage2
        import threading
        t = token["value"]
        if not t:
            return JSONResponse(status_code=401, content={"detail": "未登录"})

        def _work():
            try:
                run_stage2(cfg["server_url"], t, body.task_id,
                           cfg.get("client_id", ""),
                           rc_type=cfg.get("rc_type", "tcn"),
                           data_dir=str(DATA_DIR))
            except Exception:
                pass  # 状态已写入 failed
        threading.Thread(target=_work, daemon=True).start()
        return {"ok": True, "message": "二阶段已开始（本地训练中）"}

    @app.post("/local/stage3")
    def local_stage3(body: RcBody):
        from app.client_stage import run_stage3
        import threading
        t = token["value"]
        if not t:
            return JSONResponse(status_code=401, content={"detail": "未登录"})

        def _work():
            try:
                run_stage3(cfg["server_url"], t, body.task_id,
                           cfg.get("client_id", ""),
                           rc_type=cfg.get("rc_type", "tcn"),
                           data_dir=str(DATA_DIR))
            except Exception:
                pass
        threading.Thread(target=_work, daemon=True).start()
        return {"ok": True, "message": "本地预测生成中"}

    @app.get("/local/stage3-data")
    def local_stage3_data(task_id: int):
        from app.client_stage import get_stage3_data
        try:
            return get_stage3_data(task_id)
        except FileNotFoundError as e:
            return JSONResponse(status_code=404, content={"detail": str(e)})
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": str(e)})

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def forward(path: str, request: Request):
        method = request.method
        body = None
        if method in ("POST", "PUT"):
            body = await request.body()
        headers = {"Content-Type": "application/json"}
        # 优先透传前端携带的 Bearer（浏览器登录后请求带 token）；
        # 没有时才用本地 /local/login 存储的 token。
        auth = request.headers.get("authorization")
        t = token["value"]
        if auth:
            headers["Authorization"] = auth
            # 缓存一份，供本地 stage2/3 下载全局模型等使用（未走 /local/login 时）
            if not t:
                token["value"] = auth[7:] if auth.lower().startswith("bearer ") else auth
        elif t:
            headers["Authorization"] = "Bearer " + t
        url = cfg["server_url"] + "/api/" + path
        try:
            resp = _forward(method, url, headers, body)
        except Exception:
            return JSONResponse(status_code=502,
                                content={"detail": "远程服务器不可达"})
        if "image" in resp.headers.get("content-type", ""):
            return Response(content=resp.content,
                            media_type=resp.headers["content-type"])
        try:
            payload = resp.json()
        except Exception:
            # server 返回非 JSON（如 500 纯文本）时原样透传，避免 agent 崩溃
            return Response(
                content=resp.content,
                media_type=resp.headers.get("content-type", "text/plain"),
                status_code=resp.status_code)
        return JSONResponse(status_code=resp.status_code,
                            content=payload)

    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app


def main() -> None:
    import uvicorn
    cfg = load_config()
    port = int(cfg.get("local_port", 9001))
    app = create_app()
    print(f"FL Client Agent 启动: http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
