"""Evidence bundle: the only facts the LLM is allowed to state.

Retrieval output is flattened into atomic facts with stable ids ([E1], [E2],
...). Each fact records where it came from - a Materials Project summary
field on a Material node, or a specific graph edge (with its rule text or
similarity weight) - plus the machine-readable values the grounding checker
compares numbers against. Absent properties become explicit "not available"
facts, so the model has no gap to fill with a plausible-sounding value.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from matgraph.graph_store import Graph
from matgraph.query_schema import NUMERIC_PROPERTIES, QueryConstraints
from matgraph.retriever import Hop, RetrievalResult, node_label

FactKind = Literal["property", "edge", "missing", "similarity", "query"]

# Properties every candidate's evidence includes, beyond what the query constrains.
CORE_PROPERTIES = ["band_gap", "energy_above_hull", "formation_energy_per_atom", "density", "total_magnetization"]
OPTIONAL_PROPERTIES = [
    "bulk_modulus_vrh", "shear_modulus_vrh", "homogeneous_poisson",
    "dielectric_total", "refractive_index",
]

DFT_CAVEAT = (
    "Band gaps are DFT (GGA/GGA+U) values from Materials Project, which systematically "
    "underestimate experimental band gaps (e.g. Si: 0.61 eV computed vs 1.12 eV measured)."
)


class EvidenceFact(BaseModel):
    id: str
    kind: FactKind
    material_id: str | None = None
    formula: str | None = None
    text: str
    source: str
    values: dict[str, Any] = Field(default_factory=dict)


class EvidenceBundle(BaseModel):
    facts: list[EvidenceFact]
    material_ids: list[str]  # candidates covered, in rank order

    def by_id(self) -> dict[str, EvidenceFact]:
        return {f.id: f for f in self.facts}

    def render(self) -> str:
        """Plain-text form used in the LLM prompt."""
        lines = []
        for f in self.facts:
            lines.append(f"[{f.id}] {f.text}  (source: {f.source})")
        return "\n".join(lines)


def _fmt(prop: str, value: Any) -> str:
    unit = NUMERIC_PROPERTIES.get(prop, ("", ""))[0]
    if isinstance(value, float):
        value = round(value, 4)
    return f"{value} {unit}".strip()


def _hop_text(h: Hop) -> str:
    extra = ""
    if "rule" in h.properties:
        extra = f' (rule: "{h.properties["rule"]}")'
    elif "weight" in h.properties:
        extra = f" (weight {round(h.properties['weight'], 3)})"
    return f"{h.source} -{h.relation}-> {h.target}{extra}"


def build_evidence(
    graph: Graph,
    result: RetrievalResult,
    constraints: QueryConstraints,
    max_candidates: int = 5,
) -> EvidenceBundle:
    facts: list[EvidenceFact] = []

    def add(kind: FactKind, text: str, source: str, material_id=None, formula=None, **values) -> None:
        facts.append(EvidenceFact(
            id=f"E{len(facts) + 1}", kind=kind, material_id=material_id, formula=formula,
            text=text, source=source, values=values,
        ))

    # Query-level facts first.
    add("query", f"{result.total_matching} materials in the graph satisfy every hard constraint of the query.",
        "retrieval result", total_matching=result.total_matching)
    for prop, count in result.excluded_missing_data.items():
        add("query", f"{count} materials were excluded because Materials Project has no {prop} value for them.",
            "retrieval result", property=prop, excluded=count)
    # Always present: every candidate's evidence includes a DFT band gap.
    add("query", DFT_CAVEAT, "Materials Project methodology")

    constrained = [r.property for r in constraints.property_ranges]
    if constraints.sort_by:
        constrained.append(constraints.sort_by)

    covered = []
    for cand in result.candidates[:max_candidates]:
        props = graph.node[cand.node_id]
        mid, formula = cand.material_id, cand.formula
        covered.append(mid)
        src = f"Materials Project summary for {mid}"

        # Identity and structure.
        sg = graph.targets(cand.node_id, "has_space_group")
        sg_text = ""
        if sg:
            sg_id, sg_props = sg[0]
            cs = graph.targets(sg_id, "crystal_system")
            sg_text = f", space group {sg_props['symbol']}" + (f" ({cs[0][1]['name']} crystal system)" if cs else "")
        elements = sorted(p["symbol"] for _, p in graph.targets(cand.node_id, "contains"))
        add("edge", f"{formula} ({mid}) contains {', '.join(elements)}{sg_text}.",
            "graph edges contains / has_space_group", mid, formula, elements=elements)

        # Numeric properties: core + constrained + available optional ones.
        missing = []
        for prop in dict.fromkeys(CORE_PROPERTIES + constrained + OPTIONAL_PROPERTIES):
            value = props.get(prop)
            if value is None:
                if prop in constrained or prop in OPTIONAL_PROPERTIES:
                    missing.append(prop)
                continue
            add("property", f"{formula} {prop} = {_fmt(prop, value)}.", f"{src}, field {prop}", mid, formula,
                **{prop: value})
        if props.get("band_gap"):
            add("property", f"{formula} band gap is {'direct' if props.get('is_gap_direct') else 'indirect'}.",
                f"{src}, field is_gap_direct", mid, formula, is_gap_direct=bool(props.get("is_gap_direct")))
        ordering = props.get("ordering")
        if ordering and ordering != "Unknown":
            add("property", f"{formula} magnetic ordering = {ordering}.", f"{src}, field ordering", mid, formula,
                ordering=ordering)
        if missing:
            add("missing", f"{formula}: NOT AVAILABLE in Materials Project: {', '.join(missing)}.",
                f"{src} (fields absent)", mid, formula, missing=missing)

        # Graph edges that satisfied the query's constraints (the reasoning path).
        seen_hops = set()
        for check in cand.checks:
            for hop in check.path:
                key = (hop.source, hop.relation, hop.target)
                if key in seen_hops:
                    continue
                seen_hops.add(key)
                add("edge", _hop_text(hop), f"graph edge {hop.relation}", mid, formula, **hop.properties)
        # All application domains, not just the constrained ones.
        for domain_id, edge_props in graph.target_edges(cand.node_id, "suitable_for"):
            hop = Hop(source=node_label(props), relation="suitable_for",
                      target=node_label(graph.node[domain_id]),
                      properties={"rule": (edge_props.get("properties") or {}).get("rule")})
            key = (hop.source, hop.relation, hop.target)
            if key not in seen_hops:
                seen_hops.add(key)
                add("edge", _hop_text(hop), "graph edge suitable_for", mid, formula)

        # Similarity to anchors.
        for link in cand.anchor_links:
            for hop in link.similar_to:
                add("similarity", _hop_text(hop), "graph edge similar_to (10-d property cosine)", mid, formula,
                    weight=hop.properties.get("weight"))
            if link.shared_hubs:
                add("similarity",
                    f"{formula} shares these graph nodes with {link.anchor}: {', '.join(link.shared_hubs)}.",
                    "graph hub overlap", mid, formula, anchor=link.anchor)

    return EvidenceBundle(facts=facts, material_ids=covered)
