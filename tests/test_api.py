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


def test_ingest_requires_id_formula_or_structure(client):
    r = client.post("/materials/ingest", json={})
    assert r.status_code == 422 and "structure" in r.json()["detail"]


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
    assert body["cgcnn"]["available"] is False  # lifespan skipped in tests: models not loaded


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


# --- Phase 3: structure input ---------------------------------------------------------


@pytest.fixture
def fake_cgcnn(monkeypatch):
    from test_ingest_structure import fake_predictor

    import matgraph.ingest_structure as ist

    monkeypatch.setattr(ist, "get_predictor", lambda: fake_predictor(gap=2.0))
    monkeypatch.setattr(ist, "_stored_structure", lambda mid: ist._preview_structures.get(mid))
    ist._preview_structures.clear()


def _wurtzite_cif() -> str:
    from test_ingest_structure import wurtzite

    return wurtzite("Mg", "O").to(fmt="cif")


def test_predict_returns_predictions_without_touching_graph(client, fake_cgcnn):
    before = len(api.state["kb"].graph.node)
    r = client.post("/predict", json={"structure": _wurtzite_cif(), "filename": "MgO.cif"})
    assert r.status_code == 200
    body = r.json()
    assert body["formula"] == "MgO" and body["struct_id"].startswith("struct-MgO-sg186-")
    gap = body["predictions"]["predicted_band_gap"]
    assert gap["value"] == pytest.approx(2.0) and gap["units"] == "eV" and gap["expected_error"] > 0
    assert body["predictions"]["predicted_is_metal"]["value"] is False
    assert len(api.state["kb"].graph.node) == before


def test_predict_rejects_bad_structures(client, fake_cgcnn):
    assert client.post("/predict", json={"structure": "definitely not a crystal"}).status_code == 422
    from pymatgen.core import Lattice, Structure

    disordered = Structure(Lattice.cubic(4.0), [{"Fe": 0.5, "Ni": 0.5}], [[0, 0, 0]]).to(fmt="cif")
    r = client.post("/predict", json={"structure": disordered, "format": "cif"})
    assert r.status_code == 422 and "partial occupancies" in r.json()["detail"]


def test_predict_without_cgcnn_is_503(client, monkeypatch):
    import matgraph.ingest_structure as ist

    def missing():
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(ist, "get_predictor", missing)
    r = client.post("/predict", json={"structure": _wurtzite_cif()})
    assert r.status_code == 503 and "CGCNN unavailable" in r.json()["detail"]


def test_ingest_structure_preview_and_opt_in_retrieval(client, fake_cgcnn):
    r = client.post("/materials/ingest", json={"structure": _wurtzite_cif(), "filename": "MgO.cif",
                                               "explain": True, "use_llm": False})
    assert r.status_code == 200
    body = r.json()
    report = body["report"]
    assert report["source"] == "cgcnn" and report["material_id"].startswith("struct-MgO")
    assert body["grounding"]["traceability"] == 1.0  # templated explanation states predictions as predicted

    # The preview node exists in the graph, but queries only see it when asked to.
    question = {"question": "materials that contain Mg", "use_llm": False}
    default = client.post("/query", json=question).json()
    assert not any(c["formula"] == "MgO" for c in default["retrieval"]["candidates"])
    opted = client.post("/query", json={**question, "include_predicted": True}).json()
    mgo = next(c for c in opted["retrieval"]["candidates"] if c["formula"] == "MgO")
    assert mgo["uses_predictions"]
    row = next(row for row in opted["comparison"] if row["formula"] == "MgO")
    assert row["band_gap"] == pytest.approx(2.0) and "band_gap" in row["predicted"]
    assert opted["grounding"]["traceability"] == 1.0


def test_query_degrades_when_embeddings_fail(client, monkeypatch):
    async def broken(self, text, collections=()):
        raise RuntimeError("Ollama embedding API error: llama-server binary not found")

    monkeypatch.setattr(KnowledgeBase, "semantic_scores", broken)
    r = client.post("/query", json={"question": "battery cathodes with band gap above 3 eV", "use_llm": False})
    assert r.status_code == 200
    body = r.json()
    assert {c["formula"] for c in body["retrieval"]["candidates"]} == {"LiFePO4", "LiMnPO4"}
    assert any("Semantic search unavailable" in n for n in body["notices"])
