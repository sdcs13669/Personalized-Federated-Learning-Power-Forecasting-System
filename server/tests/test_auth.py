"""Test registration, login, JWT validation."""
import pytest
from fastapi.testclient import TestClient

from server.main import app


@pytest.fixture(autouse=True)
def setup_db(tmp_path):
    from server import database as db_mod
    from sqlalchemy.orm import sessionmaker
    db_path = str(tmp_path / "test.db")
    engine = db_mod.get_engine(f"sqlite:///{db_path}")
    db_mod.Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[db_mod.get_db] = override_get_db
    yield
    app.dependency_overrides.clear()
    engine.dispose()


client = TestClient(app)


def test_register_and_login():
    resp = client.post("/api/auth/register", json={
        "username": "alice", "password": "secret123"})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

    resp2 = client.post("/api/auth/login", json={
        "username": "alice", "password": "secret123"})
    assert resp2.status_code == 200
    assert "access_token" in resp2.json()


def test_register_duplicate():
    client.post("/api/auth/register", json={
        "username": "bob", "password": "pw1"})
    resp = client.post("/api/auth/register", json={
        "username": "bob", "password": "pw2"})
    assert resp.status_code == 409


def test_login_wrong_password():
    client.post("/api/auth/register", json={
        "username": "carol", "password": "correct"})
    resp = client.post("/api/auth/login", json={
        "username": "carol", "password": "wrong"})
    assert resp.status_code == 401


def test_me_requires_auth():
    resp = client.get("/api/me")
    assert resp.status_code == 401


def test_me_with_token():
    reg = client.post("/api/auth/register", json={
        "username": "dave", "password": "pw"})
    token = reg.json()["access_token"]
    resp = client.get("/api/me", headers={
        "Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["username"] == "dave"
