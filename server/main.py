"""FastAPI application entry point."""
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from server.database import init_db
from server.models import User
from server.routers.auth import (
    get_current_user_from_header,
    router as auth_router,
)
from server.routers.participants import router as participants_router
from server.routers.results import router as results_router
from server.routers.datasets import router as datasets_router
from server.routers.tasks import my_router, router as tasks_router

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="FL Server", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# RC 对比图的静态目录（注意：将来 B 挂 web 前端 "/" 静态目录时，必须在它之后）
RC_UPLOADS_DIR = Path(__file__).resolve().parent / "rc_uploads"
RC_UPLOADS_DIR.mkdir(exist_ok=True)
app.mount("/rc_uploads", StaticFiles(directory=str(RC_UPLOADS_DIR)),
          name="rc_uploads")


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/me")
def me(user: User = Depends(get_current_user_from_header)):
    return {"id": user.id, "username": user.username, "role": user.role}


app.include_router(auth_router)
app.include_router(tasks_router)
app.include_router(my_router)
app.include_router(participants_router)
app.include_router(results_router)
app.include_router(datasets_router)

# 前端静态文件挂载必须放在所有 /api 路由之后（优先级低），
# 未匹配 /api 的路径才落到静态文件。
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
