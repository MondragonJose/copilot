from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_all_services_up() -> None:
    with (
        patch("api.routes.health.check_database", new_callable=AsyncMock, return_value=True),
        patch("api.routes.health.check_redis", new_callable=AsyncMock, return_value=True),
    ):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"db": True, "redis": True}


def test_health_db_down() -> None:
    with (
        patch("api.routes.health.check_database", new_callable=AsyncMock, return_value=False),
        patch("api.routes.health.check_redis", new_callable=AsyncMock, return_value=True),
    ):
        resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json() == {"db": False, "redis": True}


def test_health_redis_down() -> None:
    with (
        patch("api.routes.health.check_database", new_callable=AsyncMock, return_value=True),
        patch("api.routes.health.check_redis", new_callable=AsyncMock, return_value=False),
    ):
        resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json() == {"db": True, "redis": False}


def test_health_all_down() -> None:
    with (
        patch("api.routes.health.check_database", new_callable=AsyncMock, return_value=False),
        patch("api.routes.health.check_redis", new_callable=AsyncMock, return_value=False),
    ):
        resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json() == {"db": False, "redis": False}
