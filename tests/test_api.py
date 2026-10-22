"""API contract tests against the synthetic graph (no cognee, Ollama or LLM)."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from matgraph import api
from matgraph.graph_store import KnowledgeBase


@pytest.fixture
def client(graph, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)  # never call the LLM from tests
    api.state["kb"] = KnowledgeBase(graph, None, None)
    api.state["ingest_lock"] = asyncio.Lock()
    yield TestClient(api.app)  # not a context manager: skips the cognee-loading lifespan
    api.state.clear()


def test_query_returns_grounded_answer(client):
    # LLM requested but no key configured: falls back to rules + template and says so.
    r = client.post("/query", json={"question": "battery cathodes with band gap above 3 eV", "use_llm": True})
    assert r.status_code == 200
    body = r.json()
    assert {c["formula"] for c in body["retrieval"]["candidates"]} == {"LiFePO4", "LiMnPO4"}
    assert body["grounding"]["traceability"] == 1.0
    assert body["generator"] == "template"
    assert body["comparison"][0]["band_gap"] is not None
    assert any("not configured" in n for n in body["notices"])


def test_query_validation(client):
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={"question": "ok question", "top_k": 500}).status_code == 422


def test_material_neighbourhood(client):
    r = client.get("/material/LiFePO4")
    assert r.status_code == 200
    body = r.json()
    assert body["already_in_graph"] and body["formula"] == "LiFePO4"
    assert any(e["relation"] == "similar_to" for e in body["edges"])
    assert client.get("/material/Unobtainium").status_code == 404


def test_ingest_requires_id_or_formula(client):
    r = client.post("/materials/ingest", json={})
    assert r.status_code == 422 and "CGCNN" in r.json()["detail"]


def test_ingest_existing_material_returns_relations(client):
    r = client.post("/materials/ingest", json={"formula": "GaN", "explain": True, "use_llm": False})
    assert r.status_code == 200
    body = r.json()
    assert body["report"]["already_in_graph"]
    assert body["grounding"]["traceability"] == 1.0


def test_health_reports_components(client):
    body = client.get("/health").json()
    assert body["graph_loaded"] and body["materials"] == 6
    assert "embeddings_reachable" in body and body["llm_configured"] is False


def test_cors_allows_local_origins_only(client):
    ok = client.options("/query", headers={"Origin": "null", "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "null"
    local = client.options("/query", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert local.headers.get("access-control-allow-origin") == "http://localhost:5173"
    evil = client.options("/query", headers={"Origin": "https://example.com", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in evil.headers


def test_dashboard_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "MatGraphRAG" in r.text
