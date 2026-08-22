"""Static frontend mount + CORS."""
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


def test_health_still_works():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root_serves_index():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_cors_header_present():
    r = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:9001",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.status_code == 200
