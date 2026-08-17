"""FastAPI application entry point."""
from fastapi import FastAPI, Depends

from server.database import init_db
from server.models import User
from server.routers.auth import router as auth_router, get_current_user_from_header

app = FastAPI(title="FL Server", version="0.1.0")


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/me")
def me(user: User = Depends(get_current_user_from_header)):
    return {"id": user.id, "username": user.username}


app.include_router(auth_router)
