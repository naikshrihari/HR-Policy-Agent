"""Tests for the FastAPI web UI (skipped if FastAPI isn't installed)."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hr_policy_agent.config import Settings  # noqa: E402
from hr_policy_agent.web import create_app  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("HRPA_LLM_PROVIDER", "fake")
    # Ensure a fresh agent bound to the fake provider.
    import hr_policy_agent.web as web
    web._agent.cache_clear()
    return TestClient(create_app())


def test_index_serves_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "HR Policy Assistant" in r.text


def test_config_endpoint(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    assert r.json()["provider"] == "fake"


def test_ask_returns_answer(client):
    r = client.post("/api/ask", json={"message": "How many PL days do I accrue?"})
    assert r.status_code == 200
    data = r.json()
    assert "Personal Leave" in data["response"]
    assert data["language"] == "EN"
    assert data["conversation_id"]


def test_ask_spanish(client):
    r = client.post("/api/ask", json={"message": "¿Cuándo es el día de pago?"})
    assert r.json()["language"] == "ES"
