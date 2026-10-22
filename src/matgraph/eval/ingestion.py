"""Leave-one-out evaluation of new-material ingestion (plan §7).

Hold out N materials (stratified across the fetch clusters), build the graph
without them - in memory, straight from DataPoints, no cognee writes - then
ingest each one back through ``matgraph.ingest`` and compare with the full
build:

- MP path (the cached record stands in for the API response): categorical
  edges should be recovered exactly (target 100%: same code, same data).
  similar_to is compared, as Jaccard@K, with what a full rebuild *including*
  the material would give it over the same pool - so the number isolates the
  one real design choice, z-scoring on frozen build-time statistics instead
  of recomputing them. (Comparing with the full 794-material build instead
  conflates this with held-out materials being each other's neighbours:
  stratified sampling holds out Si, Ge, GaP, SiC and BN together.)
- Composition-only path (MP lookup disabled): agreement of guessed oxidation
  states with MP's possible_species, and whether chemsys / formula pattern
  land on the same nodes.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from matgraph.enrich import (
    _cosine,
    build_graph,
    compute_similarity_stats,
    load_config,
    load_elements_table,
    load_raw_materials,
    similarity_vector,
)
from matgraph.graph_store import KnowledgeBase, graph_from_datapoints
from matgraph.ingest import CachedRecordSource, CompositionSource, ingest_material
from matgraph.retriever import node_label


def _stratified_sample(raw_all: list[dict[str, Any]], n: int, seed: int = 7) -> list[dict[str, Any]]:
    by_cluster: dict[str, list] = defaultdict(list)
    for r in raw_all:
        by_cluster[r.get("_cluster", "?")].append(r)
    rng = random.Random(seed)
    for items in by_cluster.values():
        rng.shuffle(items)
    sample, i = [], 0
    clusters = sorted(by_cluster)
    while len(sample) < n and any(by_cluster.values()):
        items = by_cluster[clusters[i % len(clusters)]]
        if items:
            sample.append(items.pop())
        i += 1
    return sample


async def run_ingestion_eval(n: int = 50) -> dict[str, Any]:
    config, elements = load_config(), load_elements_table()
    raw_all = load_raw_materials(config)
    held = _stratified_sample(raw_all, n)
    held_ids = {r["material_id"] for r in held}

    full = graph_from_datapoints(build_graph(raw_all, config, elements))
    reduced_raw = [r for r in raw_all if r["material_id"] not in held_ids]
    reduced = graph_from_datapoints(build_graph(reduced_raw, config, elements))
    stats = compute_similarity_stats(reduced_raw)
    kb = KnowledgeBase(reduced, None, None)
    top_k, min_weight = config["similarity"]["top_k"], config["similarity"]["min_edge_weight"]

    rows = []
    for raw in held:
        ref_id = full.material(raw["material_id"])
        ref_edges = {(rel, node_label(full.node[tid])) for rel, tid, _ in full.out[ref_id] if rel != "similar_to"}
        # Rebuild reference: stats recomputed with the material included.
        rebuild_stats = compute_similarity_stats(reduced_raw + [raw])
        vec = similarity_vector(raw, rebuild_stats)
        scored = sorted(((o["material_id"], _cosine(vec, similarity_vector(o, rebuild_stats))) for o in reduced_raw),
                        key=lambda pair: -pair[1])
        ref_similar = {mid for mid, w in scored[:top_k] if w >= min_weight}

        report = await ingest_material(kb, raw["material_id"], sources=[CachedRecordSource(raw)],
                                       curated=reduced_raw, stats=stats)
        got_edges = {(h.relation, h.target) for h in report.edges if h.relation != "similar_to"}
        got_similar = {n.material_id for n in report.similar_materials}
        jaccard = (len(ref_similar & got_similar) / len(ref_similar | got_similar)) if (ref_similar | got_similar) else 1.0

        comp = CompositionSource().fetch(raw["formula_pretty"])
        mp_species = set(raw.get("possible_species") or [])
        rows.append({
            "material_id": raw["material_id"], "formula": raw["formula_pretty"], "cluster": raw.get("_cluster"),
            "categorical_recovered": len(ref_edges & got_edges), "categorical_total": len(ref_edges),
            "missing_edges": sorted(map(str, ref_edges - got_edges)), "extra_edges": sorted(map(str, got_edges - ref_edges)),
            "similar_to_jaccard": round(jaccard, 4),
            "composition_pattern_match": comp["formula_anonymous"] == raw.get("formula_anonymous"),
            "composition_chemsys_match": comp["chemsys"] == raw["chemsys"],
            "oxidation_guess_exact": (set(comp["possible_species"]) == mp_species) if mp_species else None,
        })

    recovered = sum(r["categorical_recovered"] for r in rows)
    total = sum(r["categorical_total"] for r in rows)
    oxi = [r["oxidation_guess_exact"] for r in rows if r["oxidation_guess_exact"] is not None]
    return {
        "summary": {
            "held_out": len(rows),
            "mp_path_categorical_edge_recovery": round(recovered / total, 4) if total else None,
            "mp_path_similar_to_jaccard_vs_rebuild": round(sum(r["similar_to_jaccard"] for r in rows) / len(rows), 4),
            "composition_path_pattern_match": round(sum(r["composition_pattern_match"] for r in rows) / len(rows), 4),
            "composition_path_chemsys_match": round(sum(r["composition_chemsys_match"] for r in rows) / len(rows), 4),
            "composition_path_oxidation_guess_exact": round(sum(oxi) / len(oxi), 4) if oxi else None,
        },
        "rows": rows,
    }
