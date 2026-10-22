"""Explanation generation from an evidence bundle.

- ``explain_with_llm``: the LLM writes the recommendation, citing evidence
  ids for every claim (JSON-schema-constrained output).
- ``explain_with_template``: deterministic claims built directly from the
  evidence (``--no-llm`` mode and the LLM-unavailable fallback). Grounded by
  construction, so it doubles as the traceability upper bound.

Either output then goes through ``grounding.check_explanation``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from matgraph.evidence import EvidenceBundle
from matgraph.query_parser import constraints_summary
from matgraph.query_schema import QueryConstraints


class Claim(BaseModel):
    text: str = Field(description="one factual statement")
    evidence_ids: list[str] = Field(description="ids like E3 of the facts supporting it")


class Recommendation(BaseModel):
    material_id: str
    formula: str
    claims: list[Claim]


class Explanation(BaseModel):
    summary: list[Claim] = Field(description="1-3 claims answering the question overall")
    recommendations: list[Recommendation]
    caveats: list[Claim] = Field(default_factory=list)


_SYSTEM = """You are a materials-science assistant writing an evidence-backed \
recommendation. You may ONLY use the numbered facts provided. Rules:
1. Every claim must cite the ids of the facts that support it, e.g. ["E4", "E7"].
2. Recommend only materials that appear in the facts, in the given rank order.
3. Copy numbers exactly as they appear in the cited facts; never compute, \
round differently, or estimate a value.
4. If a property is marked NOT AVAILABLE, say it is not available - never guess it.
5. When band gaps matter to the question, add a caveat citing the DFT band-gap fact.
6. Keep claims short: one fact-backed statement each. Return JSON only."""
# Appended only when the evidence holds CGCNN predictions, so prompts (and the
# LLM response cache keys) for DFT-only evidence stay exactly as in Phase 2.
_PREDICTED_RULE = """
7. Facts whose source says "CGCNN prediction", and edges marked [predicted], are \
model estimates, not DFT values: call them predicted (e.g. "predicted band gap ≈ \
2.1 eV") and give their expected error when you state the number."""


def _user_prompt(question: str, constraints: QueryConstraints, evidence: EvidenceBundle) -> str:
    return (
        f"Question: {question}\n\n"
        f"Parsed constraints: {constraints_summary(constraints)}\n\n"
        f"Candidate materials (rank order): {', '.join(evidence.material_ids) or 'none'}\n\n"
        f"Facts:\n{evidence.render()}"
    )


def explain_with_llm(
    question: str, constraints: QueryConstraints, evidence: EvidenceBundle
) -> tuple[Explanation, str]:
    """Returns (explanation, model_id). Raises llm.LLMUnavailable on failure."""
    from matgraph.llm import complete_json

    messages = [
        {"role": "system", "content": _SYSTEM + (_PREDICTED_RULE if any(f.predicted for f in evidence.facts) else "")},
        {"role": "user", "content": _user_prompt(question, constraints, evidence)},
    ]
    explanation, meta = complete_json(messages, Explanation)
    return explanation, meta["model"]


def explain_with_template(evidence: EvidenceBundle) -> Explanation:
    """One claim per evidence fact, grouped by material. No LLM."""
    by_material: dict[str, list] = {}
    query_facts = []
    for fact in evidence.facts:
        if fact.material_id:
            by_material.setdefault(fact.material_id, []).append(fact)
        else:
            query_facts.append(fact)

    summary = [Claim(text=f.text, evidence_ids=[f.id]) for f in query_facts if f.values.get("total_matching") is not None]
    caveats = [Claim(text=f.text, evidence_ids=[f.id]) for f in query_facts if f.values.get("total_matching") is None]
    recommendations = []
    for mid in evidence.material_ids:
        facts = by_material.get(mid, [])
        if not facts:
            continue
        recommendations.append(Recommendation(
            material_id=mid, formula=facts[0].formula or mid,
            claims=[Claim(text=f.text, evidence_ids=[f.id]) for f in facts],
        ))
    return Explanation(summary=summary, recommendations=recommendations, caveats=caveats)
