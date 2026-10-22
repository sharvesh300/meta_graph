"""FastAPI service (PRD endpoints for Phases 2-3) + the minimal results dashboard.

    POST /query                natural-language question -> explained, evidence-backed answer
                               (include_predicted: also use CGCNN predictions; default off)
    GET  /material/{id}        stored properties + 1-hop graph neighbourhood
    POST /materials/ingest     add a material by MP id, formula or crystal structure -> relation report
    POST /predict              crystal structure -> CGCNN predictions (no graph change)
    GET  /health               readiness: graph, embeddings, LLM, CGCNN (with a fixed-structure check)
    GET  /                     dashboard (static/index.html)

Structures are sent as CIF / POSCAR text in the JSON body (the dashboard reads
the chosen file). The graph and the CGCNN models are loaded once at startup;
each request only runs retrieval, one forward pass, and the LLM when enabled.
Run with: uv run python scripts/serve.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from matgraph.answer import Answer, answer_question
from matgraph.cgcnn.validate import MAX_INPUT_BYTES
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
    state["cgcnn"] = await asyncio.to_thread(load_cgcnn)
    logger.info("CGCNN: %s", state["cgcnn"])
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


def load_cgcnn() -> dict[str, Any]:
    """Load the CGCNN models once and check them on a fixed structure. CGCNN
    is optional: without the [ml] extra or the weights, structure input is
    unavailable and everything else works."""
    try:
        from matgraph.cgcnn.predictor import weights_available
    except ImportError as error:
        return {"available": False, "reason": f"[ml] extra not installed ({error})"}
    if not weights_available():
        return {"available": False, "reason": "weights missing (run scripts/fetch_cgcnn_weights.py)"}
    from matgraph.cgcnn.registry import MODEL_SET
    from matgraph.cgcnn.smoke import smoke_test
    from matgraph.ingest_structure import get_predictor

    predictor = get_predictor()
    return {"available": True, "model_set": MODEL_SET, "models": len(predictor.models),
            "smoke_test": smoke_test(predictor)}


def kb() -> KnowledgeBase:
    if "kb" not in state:
        raise HTTPException(503, "knowledge graph not loaded")
    return state["kb"]


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    use_llm: bool = True
    top_k: int | None = Field(default=None, ge=1, le=50)
    include_predicted: bool = Field(
        default=False, description="also use CGCNN predictions (struct- materials, predicted values and edges)"
    )


StructureFormat = Literal["cif", "poscar"]


class StructureInput(BaseModel):
    structure: str = Field(min_length=10, max_length=MAX_INPUT_BYTES, description="CIF or POSCAR text")
    format: StructureFormat | None = Field(default=None, description="auto-detected when omitted")
    filename: str | None = Field(default=None, max_length=200)


class IngestRequest(BaseModel):
    material_id: str | None = Field(default=None, max_length=40)
    formula: str | None = Field(default=None, max_length=60)
    structure: str | None = Field(default=None, max_length=MAX_INPUT_BYTES, description="CIF or POSCAR text")
    structure_format: StructureFormat | None = None
    filename: str | None = Field(default=None, max_length=200)
    persist: bool = False
    explain: bool = False
    use_llm: bool = True


class PredictedValueOut(BaseModel):
    value: float | bool
    units: str
    expected_error: float | None
    expected_error_units: str
    model_ids: list[str]
    detail: dict[str, Any] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
    formula: str
    struct_id: str
    nsites: int
    space_group: str
    predictions: dict[str, PredictedValueOut]
    notes: list[str]


class IngestResponse(BaseModel):
    report: RelationReport
    explanation: dict | None = None
    grounding: dict | None = None
    generator: str | None = None
    generator_model: str | None = None


@app.post("/query", response_model=Answer)
async def query(req: QueryRequest) -> Answer:
    return await answer_question(kb(), req.question, use_llm=req.use_llm, top_k=req.top_k,
                                 include_predicted=req.include_predicted)


def _checked_structure(text: str, fmt: str | None):
    """Parse and validate an uploaded structure; 422 with the reason otherwise."""
    from matgraph.cgcnn.validate import StructureError, allowed_elements, check_structure, parse_structure
    from matgraph.enrich import load_config, load_elements_table

    try:
        structure = parse_structure(text, fmt)
        check_structure(structure, allowed_elements(load_elements_table()), set(load_config()["exclude_elements"]))
    except StructureError as error:
        raise HTTPException(422, str(error)) from error
    return structure


def _cgcnn_unavailable(error: Exception) -> HTTPException:
    return HTTPException(503, f"CGCNN unavailable: {error} (needs the [ml] extra and scripts/fetch_cgcnn_weights.py)")


@app.post("/predict", response_model=PredictResponse)
async def predict(req: StructureInput) -> PredictResponse:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    from matgraph.cgcnn.interpret import interpret
    from matgraph.cgcnn.validate import SYMPREC, struct_id
    from matgraph.ingest_structure import get_predictor

    structure = _checked_structure(req.structure, req.format)
    try:
        values = interpret(await asyncio.to_thread(lambda: get_predictor().predict(structure)))
    except (ImportError, FileNotFoundError) as error:
        raise _cgcnn_unavailable(error) from error
    sga = SpacegroupAnalyzer(structure, symprec=SYMPREC)
    return PredictResponse(
        formula=structure.composition.reduced_formula, struct_id=struct_id(structure), nsites=len(structure),
        space_group=f"{sga.get_space_group_symbol()} (#{sga.get_space_group_number()})",
        predictions={name: PredictedValueOut(
            value=v.value, units=v.units, expected_error=v.expected_error, expected_error_units=v.expected_error_units,
            model_ids=list(v.model_ids), detail=v.detail, flags=list(v.flags)) for name, v in values.items()},
        notes=["CGCNN predictions (pre-trained txie-93/cgcnn models), not DFT values. Expected errors are MAEs "
               "(error rate for is_metal) on graph materials the models had not seen; see "
               "artifacts/eval/cgcnn_graph_eval.md. Nothing was added to the graph."],
    )


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
    if not key and not req.structure:
        raise HTTPException(422, "provide material_id, formula, or structure (CIF/POSCAR text, predicted with CGCNN)")
    base = kb()
    structure = _checked_structure(req.structure, req.structure_format) if req.structure else None
    async with state["ingest_lock"]:  # overlay/graph mutation is not concurrent-safe
        try:
            if structure is not None:
                from matgraph.ingest_structure import ingest_structure

                try:
                    report = await ingest_structure(base, structure, label=req.filename or "uploaded structure",
                                                    persist=req.persist)
                except (ImportError, FileNotFoundError) as error:  # no torch / no weights
                    raise _cgcnn_unavailable(error) from error
            else:
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
    checks["cgcnn"] = state.get("cgcnn", {"available": False, "reason": "not loaded"})
    checks["status"] = "ok" if checks["graph_loaded"] and checks["embeddings_reachable"] else "degraded"
    return checks


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
