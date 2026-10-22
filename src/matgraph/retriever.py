"""Deterministic graph retrieval: QueryConstraints -> ranked candidates with
reasoning paths. No LLM anywhere in this module.

    SEED    constraint nodes (domain / class / element / species / pattern /
            space group / crystal system), anchor materials
    EXPAND  reverse traversal from each constraint node to the Materials that
            reach it (crystal systems: Material -> SpaceGroup -> CrystalSystem)
    FILTER  intersect node constraints, then enforce numeric ranges, boolean
            flags and element exclusions on Material attributes
    RANK    transparent weighted score (config/retrieval.yaml); each
            component reported per candidate
    PATHS   every satisfied constraint carries the graph path that proves it

Hard constraints are never traded off against score: a candidate satisfies
all of them or is not returned.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from matgraph.graph_store import Graph
from matgraph.query_schema import NUMERIC_PROPERTIES, QueryConstraints

ROOT = Path(__file__).resolve().parent.parent.parent
RETRIEVAL_CONFIG_PATH = ROOT / "config" / "retrieval.yaml"


def load_retrieval_config() -> dict[str, Any]:
    return yaml.safe_load(RETRIEVAL_CONFIG_PATH.read_text())


# --- result models --------------------------------------------------------------


class Hop(BaseModel):
    source: str  # node label, e.g. "Material:LiFePO4"
    relation: str
    target: str
    properties: dict[str, Any] = Field(default_factory=dict)


class ConstraintCheck(BaseModel):
    constraint: str  # human-readable, e.g. "band_gap >= 2.0 eV"
    satisfied: bool | None  # None = value missing (keep_flagged policy only)
    path: list[Hop] = Field(default_factory=list)  # graph edges proving it
    value: Any = None  # attribute value for numeric/boolean checks


class AnchorLink(BaseModel):
    anchor: str  # anchor formula
    proximity: float
    similar_to: list[Hop] = Field(default_factory=list)  # direct or 2-hop path
    shared_hubs: list[str] = Field(default_factory=list)  # most selective first


class Candidate(BaseModel):
    node_id: str
    material_id: str
    formula: str
    score: float
    score_components: dict[str, float]
    checks: list[ConstraintCheck]
    anchor_links: list[AnchorLink] = Field(default_factory=list)
    missing_properties: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    candidates: list[Candidate]
    total_matching: int  # candidates passing all hard constraints (before top_k)
    pool_size: int  # materials reached by traversal before attribute filtering
    excluded_missing_data: dict[str, int] = Field(default_factory=dict)
    mode: Literal["constrained", "anchor", "semantic", "all"]


# --- labels -------------------------------------------------------------------


def node_label(props: dict[str, Any]) -> str:
    t = props.get("type", "?")
    if t == "Material":
        return f"Material:{props['formula']} ({props['material_id']})"
    if t == "Element":
        return f"Element:{props['symbol']}"
    if t == "PropertyClass":
        return f"PropertyClass:{props['kind']}/{props['name']}"
    if t == "ApplicationDomain":
        return f"ApplicationDomain:{props['name']}"
    if t == "SpaceGroup":
        return f"SpaceGroup:{props['symbol']} (#{props['number']})"
    if t == "CrystalSystem":
        return f"CrystalSystem:{props['name']}"
    if t == "ChemicalSystem":
        return f"ChemicalSystem:{props['chemsys']}"
    if t == "OxidationState":
        return f"OxidationState:{props['species']}"
    if t == "FormulaPattern":
        return f"FormulaPattern:{props['pattern']}"
    return f"{t}:{props.get('name', '?')}"


def _hop(g: Graph, source_id: str, rel: str, target_id: str, props: dict | None = None) -> Hop:
    props = props or {}
    # cognee keeps Edge(weight=...) top-level but Edge(properties={...}) nested.
    keep = {"rule": (props.get("properties") or {}).get("rule") or props.get("rule"),
            "weight": props.get("weight")}
    keep = {k: v for k, v in keep.items() if v is not None}
    return Hop(
        source=node_label(g.node[source_id]),
        relation=rel,
        target=node_label(g.node[target_id]),
        properties=keep,
    )


def _format_range(prop: str, lo: float | None, hi: float | None) -> str:
    unit = NUMERIC_PROPERTIES[prop][0]
    unit = f" {unit}" if unit else ""
    if lo is not None and hi is not None:
        return f"{lo} <= {prop} <= {hi}{unit}"
    if lo is not None:
        return f"{prop} >= {lo}{unit}"
    return f"{prop} <= {hi}{unit}"


# --- node-constraint traversal ----------------------------------------------------


def _materials_via(g: Graph, target_id: str, rel: str) -> set[str]:
    return {sid for sid in g.sources(target_id, rel) if g.node[sid].get("type") == "Material"}


def _node_constraint_sets(
    g: Graph, c: QueryConstraints
) -> list[tuple[str, set[str], dict[str, list[Hop]]]]:
    """For each node constraint: (label, materials satisfying it, per-material proof path).

    ALL-of constraints contribute one set each; ANY-of constraints
    (crystal systems, space groups, formula patterns) contribute one set per
    field (the union of their values).
    """
    out: list[tuple[str, set[str], dict[str, list[Hop]]]] = []

    def single(label: str, target_id: str | None, rel: str) -> None:
        if target_id is None:
            out.append((label, set(), {}))
            return
        mats = _materials_via(g, target_id, rel)
        paths = {}
        for m in mats:
            edge_props = next((ep for tid, ep in g.target_edges(m, rel) if tid == target_id), {})
            paths[m] = [_hop(g, m, rel, target_id, edge_props)]
        out.append((label, mats, paths))

    for domain in c.application_domains:
        single(f"suitable for {domain}", g.find("ApplicationDomain", name=domain), "suitable_for")
    for pc in c.property_classes:
        single(f"classified as {pc.kind}/{pc.name}", g.find("PropertyClass", kind=pc.kind, name=pc.name),
               "classified_as")
    for sym in c.include_elements:
        single(f"contains {sym}", g.find("Element", symbol=sym), "contains")
    for sp in c.oxidation_states:
        single(f"has oxidation state {sp}", g.find("OxidationState", species=sp), "has_oxidation_state")

    def any_of(label: str, targets: list[str | None], rel: str) -> None:
        mats: set[str] = set()
        paths: dict[str, list[Hop]] = {}
        for target_id in targets:
            if target_id is None:
                continue
            for m in _materials_via(g, target_id, rel):
                mats.add(m)
                paths[m] = [_hop(g, m, rel, target_id)]
        out.append((label, mats, paths))

    if c.formula_patterns:
        any_of(f"formula pattern in {c.formula_patterns}",
               [g.find("FormulaPattern", pattern=p) for p in c.formula_patterns], "has_formula_pattern")
    if c.space_groups:
        sg_ids = [nid for nid, p in g.nodes_of_type("SpaceGroup") if p["symbol"] in set(c.space_groups)]
        any_of(f"space group in {c.space_groups}", sg_ids, "has_space_group")
    if c.crystal_systems:
        # Two hops: Material -has_space_group-> SpaceGroup -crystal_system-> CrystalSystem.
        mats: set[str] = set()
        paths: dict[str, list[Hop]] = {}
        for cs in c.crystal_systems:
            cs_id = g.find("CrystalSystem", name=cs)
            if cs_id is None:
                continue
            for sg_id in g.sources(cs_id, "crystal_system"):
                for m in _materials_via(g, sg_id, "has_space_group"):
                    mats.add(m)
                    paths[m] = [_hop(g, m, "has_space_group", sg_id), _hop(g, sg_id, "crystal_system", cs_id)]
        out.append((f"crystal system in {c.crystal_systems}", mats, paths))
    return out


# --- anchor proximity ---------------------------------------------------------------


class _AnchorIndex:
    """IDF-weighted hub overlap + similar_to paths, relative to anchor materials."""

    def __init__(self, g: Graph, anchor_ids: list[str], cfg: dict[str, Any]):
        self.g = g
        self.cfg = cfg["anchor"]
        self.hub_relations = set(self.cfg["hub_relations"])
        n_materials = max(1, len(g.by_type.get("Material", [])))
        self._idf_cache: dict[str, float] = {}
        self._n = n_materials
        self.anchors = [(a, self._hubs(a)) for a in anchor_ids]

    def _idf(self, hub_id: str) -> float:
        if hub_id not in self._idf_cache:
            degree = sum(1 for rel, _source, _props in self.g.in_[hub_id] if rel in self.hub_relations)
            self._idf_cache[hub_id] = math.log((self._n + 1) / (degree + 1)) + 1e-6
        return self._idf_cache[hub_id]

    def _hubs(self, material_id: str) -> set[str]:
        return {tid for rel, tid, _ in self.g.out[material_id] if rel in self.hub_relations}

    def _similar_to_path(self, candidate: str, anchor: str) -> tuple[float, list[Hop]]:
        g = self.g
        best, path = 0.0, []
        for a, b in ((candidate, anchor), (anchor, candidate)):
            for tid, ep in g.target_edges(a, "similar_to"):
                if tid == b and (ep.get("weight") or 0) > best:
                    best, path = ep["weight"], [_hop(g, a, "similar_to", b, ep)]
        discount = self.cfg["two_hop_discount"]
        for mid, ep1 in g.target_edges(candidate, "similar_to"):
            for tid, ep2 in g.target_edges(mid, "similar_to"):
                if tid == anchor:
                    w = discount * (ep1.get("weight") or 0) * (ep2.get("weight") or 0)
                    if w > best:
                        best = w
                        path = [_hop(g, candidate, "similar_to", mid, ep1), _hop(g, mid, "similar_to", anchor, ep2)]
        return best, path

    def links(self, candidate: str) -> list[AnchorLink]:
        cand_hubs = self._hubs(candidate)
        out = []
        for anchor_id, anchor_hubs in self.anchors:
            union = cand_hubs | anchor_hubs
            shared = cand_hubs & anchor_hubs
            overlap = (
                sum(self._idf(h) for h in shared) / sum(self._idf(h) for h in union) if union else 0.0
            )
            sim, path = self._similar_to_path(candidate, anchor_id)
            proximity = self.cfg["hub_weight"] * overlap + self.cfg["similarity_weight"] * sim
            shared_sorted = sorted(shared, key=lambda h: -self._idf(h))
            out.append(
                AnchorLink(
                    anchor=self.g.node[anchor_id]["formula"],
                    proximity=round(proximity, 4),
                    similar_to=path,
                    shared_hubs=[node_label(self.g.node[h]) for h in shared_sorted[:6]],
                )
            )
        return out


# --- main entry point ---------------------------------------------------------------


def _property_spread(g: Graph, prop: str) -> float:
    """Robust scale of a property: (90th - 10th percentile) / 2.56, which
    equals the standard deviation for normal data but ignores outliers."""
    values = sorted(p.get(prop) for _, p in g.nodes_of_type("Material") if p.get(prop) is not None)
    if len(values) < 2:
        return 1.0
    lo, hi = values[int(0.1 * (len(values) - 1))], values[int(0.9 * (len(values) - 1))]
    return (hi - lo) / 2.56 or 1.0


def _margin(value: float, lo: float | None, hi: float | None, spread: float) -> float:
    """0..1: how comfortably `value` sits inside [lo, hi]."""
    if lo is not None and hi is not None:
        half = (hi - lo) / 2 or spread
        return max(0.0, min(1.0, min(value - lo, hi - value) / half))
    bound = lo if lo is not None else hi
    return max(0.0, min(1.0, abs(value - bound) / spread))


def retrieve(
    graph: Graph,
    constraints: QueryConstraints,
    semantic_scores: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
    exclude_node_ids: set[str] | None = None,
) -> RetrievalResult:
    cfg = config or load_retrieval_config()
    g = graph
    c = constraints
    semantic_scores = semantic_scores or {}
    all_materials = set(g.by_type.get("Material", []))
    anchor_ids = [g.material(a) for a in c.anchor_materials]
    anchor_ids = [a for a in anchor_ids if a]
    excluded = set(anchor_ids) | (exclude_node_ids or set())

    # SEED + EXPAND: node constraints -> intersected material pool.
    node_sets = _node_constraint_sets(g, c)
    if node_sets:
        pool = set.intersection(*(s for _, s, _ in node_sets))
        mode = "constrained"
    elif anchor_ids:
        pool, mode = set(all_materials), "anchor"
    elif semantic_scores and not (c.property_ranges or c.boolean_flags or c.exclude_elements):
        pool, mode = set(all_materials), "semantic"
    else:
        pool, mode = set(all_materials), "all"
    pool -= excluded
    pool_size = len(pool)

    # FILTER: attribute-level constraints.
    policy = cfg["missing_value_policy"]
    spreads = {r.property: _property_spread(g, r.property) for r in c.property_ranges}
    excluded_missing: dict[str, int] = defaultdict(int)
    survivors: list[tuple[str, list[ConstraintCheck], list[float], list[str]]] = []

    for m in pool:
        props = g.node[m]
        checks: list[ConstraintCheck] = []
        for label, mats, paths in node_sets:
            checks.append(ConstraintCheck(constraint=label, satisfied=True, path=paths.get(m, [])))

        ok = True
        margins: list[float] = []
        missing: list[str] = []
        for r in c.property_ranges:
            value = props.get(r.property)
            label = _format_range(r.property, r.min, r.max)
            if value is None:
                if policy == "exclude":
                    excluded_missing[r.property] += 1
                    ok = False
                    break
                missing.append(r.property)
                checks.append(ConstraintCheck(constraint=label, satisfied=None, value=None))
                continue
            inside = (r.min is None or value >= r.min) and (r.max is None or value <= r.max)
            if not inside:
                ok = False
                break
            margins.append(_margin(value, r.min, r.max, spreads[r.property]))
            checks.append(ConstraintCheck(constraint=label, satisfied=True, value=value))
        if not ok:
            continue

        for b in c.boolean_flags:
            value = props.get(b.flag)
            if value is None or bool(value) != b.value:
                ok = False
                break
            checks.append(ConstraintCheck(constraint=f"{b.flag} = {b.value}", satisfied=True, value=value))
        if not ok:
            continue

        if c.exclude_elements:
            present = {p["symbol"] for _, p in g.targets(m, "contains")}
            if present & set(c.exclude_elements):
                continue
            checks.append(ConstraintCheck(constraint=f"contains none of {c.exclude_elements}", satisfied=True,
                                          value=sorted(present)))
        survivors.append((m, checks, margins, missing))

    # RANK.
    weights = cfg["ranking_weights"]
    anchor_index = _AnchorIndex(g, anchor_ids, cfg) if anchor_ids else None
    sem_values = [semantic_scores[m] for m, *_ in survivors if m in semantic_scores]
    sem_lo, sem_hi = (min(sem_values), max(sem_values)) if sem_values else (0.0, 1.0)

    candidates: list[Candidate] = []
    for m, checks, margins, missing in survivors:
        props = g.node[m]
        components: dict[str, float] = {}
        if margins:
            components["constraint_margin"] = sum(margins) / len(margins)
        links: list[AnchorLink] = []
        if anchor_index:
            links = anchor_index.links(m)
            components["anchor"] = max(link.proximity for link in links)
        if m in semantic_scores and sem_hi > sem_lo:
            components["semantic"] = (semantic_scores[m] - sem_lo) / (sem_hi - sem_lo)
        e_hull = props.get("energy_above_hull")
        if e_hull is not None:
            components["stability"] = 1.0 / (1.0 + max(0.0, e_hull) / cfg["stability_scale"])
        total_weight = sum(weights[k] for k in components)
        score = sum(weights[k] * v for k, v in components.items()) / total_weight if total_weight else 0.0
        candidates.append(
            Candidate(
                node_id=m,
                material_id=props["material_id"],
                formula=props["formula"],
                score=round(score, 4),
                score_components={k: round(v, 4) for k, v in components.items()},
                checks=checks,
                anchor_links=links,
                missing_properties=missing,
            )
        )

    def sort_key(cand: Candidate):
        incomplete = 1 if cand.missing_properties else 0
        if c.sort_by:
            value = g.node[cand.node_id].get(c.sort_by)
            has_value = 0 if value is not None else 1
            ordered = -(value or 0) if c.sort_descending else (value or 0)
            return (incomplete, has_value, ordered, -cand.score, cand.formula)
        return (incomplete, -cand.score, cand.formula)

    candidates.sort(key=sort_key)
    return RetrievalResult(
        candidates=candidates[: c.top_k],
        total_matching=len(candidates),
        pool_size=pool_size,
        excluded_missing_data=dict(excluded_missing),
        mode=mode,
    )
