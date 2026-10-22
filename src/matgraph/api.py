"""FastAPI service (PRD endpoints for Phase 2) + the minimal results dashboard.

    POST /query                natural-language question -> explained, evidence-backed answer
    GET  /material/{id}        stored properties + 1-hop graph neighbourhood
    POST /materials/ingest     add a material by MP id or formula -> relation report
    GET  /health               readiness: graph, vector store, embeddings, LLM
    GET  /                     dashboard (static/index.html)

The graph is loaded once at startup; each request only runs retrieval (and
the LLM, when enabled). Run with: uv run python scripts/serve.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from matgraph.answer import Answer, answer_question
from matgraph.graph_store import KnowledgeBase, load_knowledge_base
from matgraph.ingest import IngestError, RelationReport, explain_relation, ingest_material, relation_report_for_existing
from matgraph.llm import llm_configured

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["kb"] = await load_knowledge_base()
    state["ingest_lock"] = asyncio.Lock()
    logger.info("knowledge graph loaded: %d nodes", len(state["kb"].graph.node))
    yield
    state.clear()


app = FastAPI(title="MatGraphRAG", version="0.2.0", lifespan=lifespan)

# The dashboard is normally served from this app (same origin). CORS only
# covers opening static/index.html directly (origin "null") or from another
# local port - never arbitrary websites, since /materials/ingest can write.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def kb() -> KnowledgeBase:
    if "kb" not in state:
        raise HTTPException(503, "knowledge graph not loaded")
    return state["kb"]


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    use_llm: bool = True
    top_k: int | None = Field(default=None, ge=1, le=50)


class IngestRequest(BaseModel):
    material_id: str | None = Field(default=None, max_length=40)
    formula: str | None = Field(default=None, max_length=60)
    persist: bool = False
    explain: bool = False
    use_llm: bool = True


class IngestResponse(BaseModel):
    report: RelationReport
    explanation: dict | None = None
    grounding: dict | None = None
    generator: str | None = None
    generator_model: str | None = None


@app.post("/query", response_model=Answer)
async def query(req: QueryRequest) -> Answer:
    return await answer_question(kb(), req.question, use_llm=req.use_llm, top_k=req.top_k)


@app.get("/material/{material_id}", response_model=RelationReport)
async def material(material_id: str) -> RelationReport:
    base = kb()
    node_id = base.graph.material(material_id)
    if node_id is None:
        raise HTTPException(404, f"{material_id!r} is not in the graph (add it with POST /materials/ingest)")
    return relation_report_for_existing(base, node_id, material_id)


@app.post("/materials/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    key = (req.material_id or req.formula or "").strip()
    if not key:
        raise HTTPException(422, "provide material_id or formula (crystal structures go to CGCNN in Phase 3)")
    base = kb()
    async with state["ingest_lock"]:  # overlay/graph mutation is not concurrent-safe
        try:
            report = await ingest_material(base, key, persist=req.persist)
        except IngestError as error:
            raise HTTPException(422, str(error)) from error
    response = IngestResponse(report=report)
    if req.explain:
        explanation, grounding, generator, model = await asyncio.to_thread(
            explain_relation, base, report, req.use_llm
        )
        response.explanation = explanation.model_dump()
        response.grounding = grounding.model_dump()
        response.generator, response.generator_model = generator, model
    return response


@app.get("/health")
async def health() -> dict[str, Any]:
    checks: dict[str, Any] = {"graph_loaded": "kb" in state}
    if "kb" in state:
        g = state["kb"].graph
        checks["materials"] = len(g.by_type.get("Material", []))
        checks["preview_materials"] = sum(1 for n in g.overlay_nodes if g.node[n].get("type") == "Material")
    endpoint = os.getenv("EMBEDDING_ENDPOINT", "http://localhost:11434/api/embed")
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            base = endpoint.split("/api/")[0]
            checks["embeddings_reachable"] = (await client.get(f"{base}/api/tags")).status_code == 200
    except httpx.HTTPError:
        checks["embeddings_reachable"] = False
    checks["llm_configured"] = llm_configured()
    checks["status"] = "ok" if checks["graph_loaded"] and checks["embeddings_reachable"] else "degraded"
    return checks


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
