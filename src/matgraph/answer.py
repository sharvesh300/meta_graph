"""End-to-end question answering: parse -> retrieve -> evidence -> explain -> ground.

Shared by the CLI (scripts/ask.py) and the API (matgraph.api).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from matgraph.evidence import CORE_PROPERTIES, OPTIONAL_PROPERTIES, EvidenceBundle, build_evidence
from matgraph.generator import Explanation, explain_with_llm, explain_with_template
from matgraph.graph_store import KnowledgeBase
from matgraph.grounding import GroundingReport, check_explanation
from matgraph.llm import LLMUnavailable, llm_configured
from matgraph.query_parser import parse_question
from matgraph.query_schema import ParsedQuery
from matgraph.retriever import RetrievalResult, load_retrieval_config, retrieve

logger = logging.getLogger(__name__)


class Answer(BaseModel):
    question: str
    parsed: ParsedQuery
    retrieval: RetrievalResult
    comparison: list[dict[str, Any]]  # property table of the top candidates (None = not available)
    evidence: EvidenceBundle
    explanation: Explanation
    grounding: GroundingReport
    generator: Literal["llm", "template"]
    generator_model: str | None = None
    notices: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)


def comparison_table(kb: KnowledgeBase, result: RetrievalResult, extra: list[str]) -> list[dict[str, Any]]:
    columns = list(dict.fromkeys(CORE_PROPERTIES + extra + OPTIONAL_PROPERTIES))
    rows = []
    for cand in result.candidates:
        props = kb.graph.node[cand.node_id]
        row = {"material_id": cand.material_id, "formula": cand.formula, "score": cand.score}
        row.update({col: props.get(col) for col in columns})
        rows.append(row)
    return rows


async def answer_question(
    kb: KnowledgeBase,
    question: str,
    use_llm: bool = True,
    top_k: int | None = None,
    use_semantic: bool = True,
    max_explained: int = 5,
) -> Answer:
    timings: dict[str, float] = {}
    notices: list[str] = []
    llm_on = use_llm and llm_configured()
    if use_llm and not llm_configured():
        notices.append("LLM not configured (OPENROUTER_API_KEY); used rule-based parsing and templated explanation.")

    t = time.perf_counter()
    # LLM calls are blocking HTTP requests: keep them off the event loop.
    parsed = await asyncio.to_thread(parse_question, question, kb.graph, llm_on)
    if llm_on and parsed.parser == "rules":
        notices.append("LLM unavailable for parsing (rate limit or outage); used the rule-based parser.")
    if top_k:
        parsed.constraints.top_k = top_k
    timings["parse"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    cfg = load_retrieval_config()
    semantic = {}
    if use_semantic:
        semantic = await kb.semantic_scores(parsed.constraints.semantic_text, tuple(cfg["semantic_collections"]))
    timings["semantic"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    result = retrieve(kb.graph, parsed.constraints, semantic, cfg)
    timings["retrieve"] = (time.perf_counter() - t) * 1000

    evidence = build_evidence(kb.graph, result, parsed.constraints, max_candidates=max_explained)
    constrained = [r.property for r in parsed.constraints.property_ranges]
    if parsed.constraints.sort_by:
        constrained.append(parsed.constraints.sort_by)

    t = time.perf_counter()
    generator, model = "template", None
    explanation: Explanation
    if llm_on and evidence.material_ids:
        try:
            explanation, model = await asyncio.to_thread(
                explain_with_llm, question, parsed.constraints, evidence
            )
            generator = "llm"
        except LLMUnavailable as error:
            logger.warning("LLM explanation unavailable: %s", error)
            notices.append("LLM unavailable for the explanation; showing the templated, evidence-only explanation.")
            explanation = explain_with_template(evidence)
    else:
        explanation = explain_with_template(evidence)
    timings["explain"] = (time.perf_counter() - t) * 1000

    constraint_numbers: list[Any] = [parsed.constraints.top_k]
    for r in parsed.constraints.property_ranges:
        constraint_numbers += [r.min, r.max]
    grounding = check_explanation(
        explanation, evidence, kb.graph, [c.material_id for c in result.candidates], question, constraint_numbers
    )
    if not result.candidates:
        notices.append("No material in the graph satisfies every constraint.")

    return Answer(
        question=question, parsed=parsed, retrieval=result,
        comparison=comparison_table(kb, result, constrained), evidence=evidence,
        explanation=explanation, grounding=grounding, generator=generator, generator_model=model,
        notices=notices, timings_ms={k: round(v, 1) for k, v in timings.items()},
    )
