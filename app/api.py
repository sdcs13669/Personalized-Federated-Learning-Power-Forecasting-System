"""后端接口封装：登录 / 注册 / 任务广场 / 发起任务 / 审计。

用法：from app import api
    token = api.login("用户名", "密码")        # 成功返回 access_token
    tasks = api.get_tasks(token)              # 返回任务列表
后端地址：默认 http://localhost:8000（改 BASE 即可）。
"""
from __future__ import annotations

import requests

BASE = "http://localhost:8000"


class ApiError(Exception):
    """接口调用失败，message 是可直接展示给用户的中文提示。"""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _handle(resp: requests.Response) -> dict | list:
    """把后端返回解析成 JSON；出错时抛 ApiError。"""
    try:
        data = resp.json()
    except Exception:
        data = {}
    if resp.status_code >= 400:
        detail = data.get("detail", f"服务器返回错误（{resp.status_code}）")
        raise ApiError(str(detail), resp.status_code)
    return data


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def register(username: str, password: str) -> str:
    """注册新账号，成功返回 access_token。"""
    resp = requests.post(f"{BASE}/api/auth/register",
                         json={"username": username, "password": password},
                         timeout=10)
    return _handle(resp)["access_token"]


def login(username: str, password: str) -> str:
    """登录，成功返回 access_token。"""
    resp = requests.post(f"{BASE}/api/auth/login",
                         json={"username": username, "password": password},
                         timeout=10)
    return _handle(resp)["access_token"]


def get_tasks(token: str) -> list:
    """任务广场：获取所有招募中的任务。"""
    resp = requests.get(f"{BASE}/api/tasks",
                        headers=_auth(token), timeout=10)
    return _handle(resp)


def create_task(token: str, name: str, rounds: int) -> dict:
    """发起新任务，返回任务字典（含参与密钥 key）。"""
    resp = requests.post(f"{BASE}/api/tasks",
                         headers=_auth(token),
                         json={"name": name, "rounds": rounds},
                         timeout=10)
    return _handle(resp)


def get_audit(token: str, task_id: int) -> list:
    """获取某任务的逐轮审计记录（参与/掉线/损失）。"""
    resp = requests.get(f"{BASE}/api/tasks/{task_id}/audit",
                        headers=_auth(token), timeout=10)
    return _handle(resp)
