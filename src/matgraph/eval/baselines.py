"""Retrieval configurations compared in the evaluation (plan §7).

All return a ranked list of material_ids. The LLM, prompt and K are held
fixed across configurations; only the retrieval mechanism varies.
"""

from __future__ import annotations

import copy
from typing import Any

from matgraph.graph_store import Graph
from matgraph.query_schema import QueryConstraints
from matgraph.retriever import retrieve


def vector_rag(graph: Graph, semantic: dict[str, float], k: int) -> list[str]:
    """B1: rank every material by embedding similarity of the question to its
    profile/description text. No graph, no constraint enforcement."""
    materials = [(nid, semantic.get(nid, 0.0)) for nid in graph.by_type.get("Material", [])]
    materials.sort(key=lambda pair: -pair[1])
    return [graph.node[nid]["material_id"] for nid, _ in materials[:k]]


def _seed_nodes(graph: Graph, c: QueryConstraints) -> list[str]:
    seeds = []
    for d in c.application_domains:
        seeds.append(graph.find("ApplicationDomain", name=d))
    for pc in c.property_classes:
        seeds.append(graph.find("PropertyClass", kind=pc.kind, name=pc.name))
    for e in c.include_elements:
        seeds.append(graph.find("Element", symbol=e))
    for s in c.oxidation_states:
        seeds.append(graph.find("OxidationState", species=s))
    for p in c.formula_patterns:
        seeds.append(graph.find("FormulaPattern", pattern=p))
    for sg in c.space_groups:
        seeds.append(graph.find("SpaceGroup", symbol=sg))
    for cs in c.crystal_systems:
        seeds.append(graph.find("CrystalSystem", name=cs))
    for a in c.anchor_materials:
        anchor = graph.material(a)
        if anchor:
            seeds.append(anchor)
    return [s for s in seeds if s]


def graphrag_fixed_hop(graph: Graph, c: QueryConstraints, semantic: dict[str, float], k: int) -> list[str]:
    """B2: generic GraphRAG. Seed nodes are the entities named in the query;
    expand a fixed neighbourhood (1 hop; 2 for crystal systems, which sit
    behind SpaceGroup); rank materials by how many seeds they touch, ties by
    embedding similarity. No attribute filtering: numeric ranges, boolean
    flags and exclusions are not graph entities, so this baseline cannot
    enforce them - which is exactly what it measures."""
    touches: dict[str, int] = {}
    anchors = {graph.material(a) for a in c.anchor_materials}
    for seed in _seed_nodes(graph, c):
        reached: set[str] = set()
        seed_type = graph.node[seed].get("type")
        if seed_type == "Material":
            # Anchor: materials one hop away (similar_to), plus materials
            # sharing any of its hub nodes (Material -> hub <- Material).
            for rel, tid, _ in graph.out[seed]:
                if rel == "similar_to":
                    reached.add(tid)
                else:
                    reached.update(s for r, s, _ in graph.in_[tid] if graph.node[s].get("type") == "Material")
        elif seed_type == "CrystalSystem":
            for sg in graph.sources(seed, "crystal_system"):
                reached.update(graph.sources(sg, "has_space_group"))
        else:
            reached.update(s for _, s, _ in graph.in_[seed] if graph.node[s].get("type") == "Material")
        for m in reached - anchors:
            touches[m] = touches.get(m, 0) + 1
    if not touches:  # nothing nameable in the query: fall back to embeddings
        return vector_rag(graph, semantic, k)
    ranked = sorted(touches, key=lambda m: (-touches[m], -semantic.get(m, 0.0)))
    return [graph.node[m]["material_id"] for m in ranked[:k]]


def proposed(graph: Graph, c: QueryConstraints, semantic: dict[str, float], cfg: dict[str, Any], k: int) -> list[str]:
    constraints = c.model_copy(update={"top_k": k})
    return [cand.material_id for cand in retrieve(graph, constraints, semantic, cfg).candidates]


def proposed_no_similarity(graph: Graph, c: QueryConstraints, semantic: dict[str, float], cfg: dict[str, Any], k: int) -> list[str]:
    """Ablation: anchor proximity from shared hubs only (no similar_to edges)."""
    cfg = copy.deepcopy(cfg)
    cfg["anchor"]["similarity_weight"] = 0.0
    cfg["anchor"]["hub_weight"] = 1.0
    return proposed(graph, c, semantic, cfg, k)
