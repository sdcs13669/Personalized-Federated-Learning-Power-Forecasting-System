"""client_agent：本地 HTTP 代理（静态前端 + REST 转发 + 本地控制端点）。

启动: python app/agent.py
浏览器打开 http://localhost:9001
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEB = ROOT / "web"
CONFIG_PATH = Path(__file__).resolve().parent / "agent_config.json"
DATA_DIR = Path(__file__).resolve().parent / "data"

_default_config = {
    "server_url": "http://127.0.0.1:8000",
    "username": "",
    "password": "",
    "client_id": "",
    "local_port": 9001,
}


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
    try:
        resp = urllib.request.urlopen(req, timeout=10)
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


class LoginBody(BaseModel):
    username: str
    password: str


def create_app(web_dir: str | None = None,
               server_url: str | None = None,
               config_path: Path | None = None) -> FastAPI:
    web_dir = web_dir or str(DEFAULT_WEB)
    cfg = load_config()
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

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def forward(path: str, request: Request):
        method = request.method
        body = None
        if method in ("POST", "PUT"):
            body = await request.body()
        headers = {"Content-Type": "application/json"}
        t = token["value"]
        if t:
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
        return JSONResponse(status_code=resp.status_code,
                            content=resp.json())

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
