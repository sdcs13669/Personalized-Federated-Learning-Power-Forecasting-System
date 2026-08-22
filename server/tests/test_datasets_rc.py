"""datasets 清单 + RC 结果上传/查询。"""
import hashlib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from server.main import app
from server import database as db_mod
import server.routers.results as results_mod


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    engine = db_mod.get_engine(f"sqlite:///{db_path}")
    db_mod.Base.metadata.create_all(bind=engine)
    db_mod._migrate(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    # png 写到临时目录，避免污染仓库
    monkeypatch.setattr(results_mod, "RC_UPLOADS_DIR", tmp_path / "rc_uploads")

    def override_get_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[db_mod.get_db] = override_get_db
    yield
    app.dependency_overrides.clear()
    engine.dispose()


client = TestClient(app)

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 100


def _register(name):
    r = client.post("/api/auth/register",
                    json={"username": name, "password": "pw123456"})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["access_token"]}


def _create_and_join(name):
    h = _register(name)
    r = client.post("/api/tasks", headers=h, json={"name": "t", "rounds": 3})
    assert r.status_code == 200, r.text
    task_id = r.json()["id"]
    key_hash = hashlib.sha256(r.json()["key"].encode()).hexdigest()
    r = client.post(f"/api/tasks/{task_id}/join", headers=h,
                    json={"key_hash": key_hash, "client_id": "steel_ind_0"})
    assert r.status_code == 200, r.text
    return h, task_id


def test_datasets_list():
    h = _register("u1")
    r = client.get("/api/datasets", headers=h)
    assert r.status_code == 200
    items = r.json()
    ids = {d["id"] for d in items}
    assert {"steel_ind_0", "tetouan_0", "tetouan_1", "tetouan_2"} <= ids
    steel = next(d for d in items if d["id"] == "steel_ind_0")
    assert steel["url"].startswith("https://")
    assert steel["client_id"] == "steel_ind_0"
    # client_id 必须与 app/agent.py 对齐：tetouan 是 tetouan_city_*，不是 tetouan_*
    by_id = {d["id"]: d["client_id"] for d in items}
    assert by_id["tetouan_0"] == "tetouan_city_0"
    assert by_id["tetouan_1"] == "tetouan_city_1"
    assert by_id["tetouan_2"] == "tetouan_city_2"


def test_rc_result_roundtrip():
    h, task_id = _create_and_join("u2")
    files = {"png": ("cmp.png", PNG_BYTES, "image/png")}
    r = client.post(f"/api/tasks/{task_id}/rc-result", headers=h,
                    data={"client_id": "steel_ind_0",
                          "wape_global": "12.3", "wape_rc": "9.8"},
                    files=files)
    assert r.status_code == 200, r.text

    r = client.get(f"/api/tasks/{task_id}/rc-results", headers=h)
    assert r.status_code == 200
    results = r.json()
    assert len(results) == 1
    assert results[0]["wape_rc"] == 9.8
    assert results[0]["png_url"].startswith("/rc_uploads/")


def test_rc_result_requires_participant():
    h = _register("u3")
    h2 = _register("u4")
    r = client.post("/api/tasks", headers=h, json={"name": "t2", "rounds": 3})
    task_id = r.json()["id"]
    files = {"png": ("cmp.png", PNG_BYTES, "image/png")}
    r = client.post(f"/api/tasks/{task_id}/rc-result", headers=h2,
                    data={"client_id": "steel_ind_0",
                          "wape_global": "1", "wape_rc": "1"},
                    files=files)
    assert r.status_code == 403


def test_rc_result_overwrite():
    h, task_id = _create_and_join("u5")

    def upload(wape_global, wape_rc):
        files = {"png": ("cmp.png", PNG_BYTES, "image/png")}
        r = client.post(f"/api/tasks/{task_id}/rc-result", headers=h,
                        data={"client_id": "steel_ind_0",
                              "wape_global": str(wape_global),
                              "wape_rc": str(wape_rc)},
                        files=files)
        assert r.status_code == 200, r.text

    upload(10.0, 9.0)
    upload(8.0, 7.0)
    r = client.get(f"/api/tasks/{task_id}/rc-results", headers=h)
    results = r.json()
    assert len(results) == 1
    assert results[0]["wape_global"] == 8.0
    assert results[0]["wape_rc"] == 7.0
