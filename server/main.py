"""FastAPI application entry point."""
from fastapi import FastAPI

from server.database import init_db

app = FastAPI(title="FL Server", version="0.1.0")


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/api/health")
def health():
    return {"status": "ok"}
