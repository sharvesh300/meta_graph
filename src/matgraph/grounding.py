"""Deterministic check that an explanation only states what the evidence says.

A claim is grounded when:
1. it cites at least one evidence id, and every cited id exists;
2. every number in its text matches a value in the cited facts (or in the
   user's own question/constraints, e.g. "above the requested 2 eV") within
   rounding tolerance;
3. every material formula it mentions is one covered by the evidence;
4. (recommendations) the recommended material is one that was retrieved;
5. a claim that rests on a CGCNN prediction - it states a number only a
   predicted fact supports, or cites nothing but predicted facts - says so
   ("predicted", "estimated", "≈", ...): a model estimate must never read
   as a DFT value.

The traceability score - the PRD's "proportion of statements traceable to a
graph edge or database record" - is grounded claims / all claims. Ungrounded
claims are returned flagged, never hidden.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from matgraph.evidence import EvidenceBundle, EvidenceFact
from matgraph.generator import Claim, Explanation
from matgraph.graph_store import Graph

# Numbers that are part of identifiers, not quantities: formula subscripts
# (LiFePO4), oxidation states (Fe2+), fact ids (E12), material ids
# (mp-1234), space group symbols (P6_3mc) and patterns (ABC3).
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_\-+()./]*|-?\d+(?:\.\d+)?(?:[eE]-?\d+)?")
_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE]-?\d+)?$")
_FORMULA = re.compile(r"\b(?:[A-Z][a-z]?\d*(?:\.\d+)?|\((?:[A-Z][a-z]?\d*)+\)\d*){2,}\b")
_HEDGE = re.compile(r"predict|estimat|≈|~|approx|cgcnn|model", re.IGNORECASE)


class ClaimCheck(BaseModel):
    section: str  # "summary" | "recommendation:<material_id>" | "caveat"
    text: str
    evidence_ids: list[str]
    grounded: bool
    problems: list[str] = Field(default_factory=list)


class GroundingReport(BaseModel):
    claims: list[ClaimCheck]
    traceability: float
    grounded_claims: int
    total_claims: int
    unretrieved_recommendations: list[str] = Field(default_factory=list)


def _numbers_in(text: str) -> list[tuple[float, int]]:
    """Standalone numeric quantities in `text` as (value, decimals written),
    excluding digits inside identifiers."""
    out = []
    for token in _TOKEN.findall(text):
        if _NUMBER.match(token):
            mantissa = token.lstrip("-").split("e")[0].split("E")[0]
            decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
            out.append((float(token), decimals))
    return out


def _fact_numbers(fact: EvidenceFact) -> list[float]:
    nums = [float(v) for v in fact.values.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    nums += [n for n, _ in _numbers_in(fact.text)]
    return nums


def _matches(value: float, decimals: int, candidates: list[float]) -> bool:
    """True if some candidate rounds to `value` at the precision it was
    written with, or is within 2%. Compared by magnitude, because "1.5-3.5"
    tokenizes as 1.5 and -3.5."""
    tolerance = 0.5 * 10 ** (-decimals) + 1e-9
    v = abs(value)
    return any(abs(v - abs(c)) <= max(tolerance, 0.02 * abs(c)) for c in candidates)


def check_claim(
    claim: Claim,
    section: str,
    facts: dict[str, EvidenceFact],
    allowed_numbers: list[float],
    evidence_formulas: set[str],
    graph_formulas: set[str],
) -> ClaimCheck:
    problems = []
    cited = [facts[i] for i in claim.evidence_ids if i in facts]
    unknown = [i for i in claim.evidence_ids if i not in facts]
    if not claim.evidence_ids:
        problems.append("no evidence cited")
    if unknown:
        problems.append(f"cites unknown evidence ids {unknown}")

    supported = [n for f in cited for n in _fact_numbers(f)] + allowed_numbers
    exact = [n for f in cited if not f.predicted for n in _fact_numbers(f)] + allowed_numbers
    rests_on_prediction = bool(cited) and all(f.predicted for f in cited)
    for number, decimals in _numbers_in(claim.text):
        if not _matches(number, decimals, supported):
            problems.append(f"number {number:g} not found in cited evidence")
        elif not _matches(number, decimals, exact):
            rests_on_prediction = True
    if rests_on_prediction and not _HEDGE.search(claim.text):
        problems.append("states a CGCNN prediction without saying it is predicted")

    for formula in set(_FORMULA.findall(claim.text)):
        if formula in graph_formulas and formula not in evidence_formulas:
            problems.append(f"mentions {formula}, which is not in the evidence")

    return ClaimCheck(section=section, text=claim.text, evidence_ids=claim.evidence_ids,
                      grounded=not problems, problems=problems)


def check_explanation(
    explanation: Explanation,
    evidence: EvidenceBundle,
    graph: Graph,
    retrieved_material_ids: list[str],
    question: str,
    constraint_numbers: list[Any] | None = None,
) -> GroundingReport:
    facts = evidence.by_id()
    allowed = [n for n, _ in _numbers_in(question)] + [float(n) for n in (constraint_numbers or []) if n is not None]
    evidence_formulas = {f.formula for f in evidence.facts if f.formula}
    # Anchors and shared-hub text may name other materials; those are in fact text.
    for f in evidence.facts:
        evidence_formulas.update(_FORMULA.findall(f.text))
    graph_formulas = set(graph.material_by_formula)

    checks: list[ClaimCheck] = []
    for claim in explanation.summary:
        checks.append(check_claim(claim, "summary", facts, allowed, evidence_formulas, graph_formulas))
    unretrieved = []
    for rec in explanation.recommendations:
        section = f"recommendation:{rec.material_id}"
        not_retrieved = rec.material_id not in retrieved_material_ids
        if not_retrieved:
            unretrieved.append(rec.material_id)
        for claim in rec.claims:
            check = check_claim(claim, section, facts, allowed, evidence_formulas, graph_formulas)
            if not_retrieved:
                check.problems.append(f"{rec.material_id} was not retrieved")
                check.grounded = False
            checks.append(check)
    for claim in explanation.caveats:
        checks.append(check_claim(claim, "caveat", facts, allowed, evidence_formulas, graph_formulas))

    grounded = sum(1 for c in checks if c.grounded)
    return GroundingReport(
        claims=checks,
        traceability=round(grounded / len(checks), 4) if checks else 1.0,
        grounded_claims=grounded,
        total_claims=len(checks),
        unretrieved_recommendations=unretrieved,
    )
