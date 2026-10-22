"""Structural verification of the built graph.

Deliberately LLM-free: Phase 1's deliverable is the graph itself, so
correctness is checked by pulling the raw node/edge data straight from the
graph engine (``get_graph_data``) and inspecting it in Python, rather than
through cognee's natural-language search (which needs an LLM_API_KEY that
this environment does not have configured) or hand-written Cypher (cognee's
Kuzu/Ladybug backend stores every node/edge in generic ``Node``/``EDGE``
tables with a JSON properties blob, not per-class labels/columns).
"""

from __future__ import annotations

import logging
from collections import defaultdict

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.users.methods import get_default_user

from matgraph.graph_store import Graph
from matgraph.pipeline import DATASET_NAME

logger = logging.getLogger(__name__)


async def verify() -> None:
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets([DATASET_NAME], "read", user)

    async with set_database_global_context_variables(datasets[0].id, user.id):
        engine = await get_graph_engine()
        raw_nodes, raw_edges = await engine.get_graph_data()
        g = Graph(raw_nodes, raw_edges)

        print("\n== Node counts by type ==")
        by_type: dict[str, int] = defaultdict(int)
        for _, props in g.node.items():
            by_type[props.get("type", "?")] += 1
        for type_name, count in sorted(by_type.items(), key=lambda kv: -kv[1]):
            print(f"  {type_name:20s} {count}")

        print("\n== Edge counts by relationship ==")
        by_rel: dict[str, int] = defaultdict(int)
        for _, _, rel, _ in raw_edges:
            by_rel[rel] += 1
        for rel, count in sorted(by_rel.items(), key=lambda kv: -kv[1]):
            print(f"  {rel:20s} {count}")

        materials = {p["formula"]: (nid, p) for nid, p in g.nodes_of_type("Material")}
        sample_cap = 15  # print at most this many examples per section; always show the count

        def print_sample(rows: list[str], label: str) -> None:
            print(f"  {len(rows)} {label}")
            for row in rows[:sample_cap]:
                print(f"    {row}")
            if len(rows) > sample_cap:
                print(f"    ... and {len(rows) - sample_cap} more")

        print("\n== Oxygen hub: materials containing O ==")
        oxygen_materials = [
            formula
            for formula, (nid, _) in sorted(materials.items())
            if "O" in {ep["symbol"] for _, ep in g.targets(nid, "contains")}
        ]
        print_sample(oxygen_materials, "materials contain O")

        print("\n== Lithium-containing materials classified as 'stable' ==")
        li_stable = []
        for formula, (nid, _) in sorted(materials.items()):
            elements = {ep["symbol"] for _, ep in g.targets(nid, "contains")}
            classes = {(cp["kind"], cp["name"]) for _, cp in g.targets(nid, "classified_as")}
            if "Li" in elements and ("stability", "stable") in classes:
                li_stable.append(formula)
        print_sample(li_stable, "Li-containing materials classified as stable")

        print("\n== Wide-band-gap materials ==")
        rows = []
        for formula, (nid, p) in materials.items():
            classes = {(cp["kind"], cp["name"]) for _, cp in g.targets(nid, "classified_as")}
            if ("band_gap", "wide_gap") in classes:
                rows.append((formula, p["band_gap"]))
        rows.sort(key=lambda r: -r[1])
        print_sample([f"{formula:12s} band_gap={bg:.3f}" for formula, bg in rows], "wide-gap materials")

        if "LiFePO4" in materials and "Fe2O3" in materials:
            print("\n== What connects LiFePO4 and Fe2O3 (shared elements) ==")
            nid_a, _ = materials["LiFePO4"]
            nid_b, _ = materials["Fe2O3"]
            elems_a = {ep["symbol"] for _, ep in g.targets(nid_a, "contains")}
            elems_b = {ep["symbol"] for _, ep in g.targets(nid_b, "contains")}
            for symbol in sorted(elems_a & elems_b):
                print(f"  shared element: {symbol}")

        print("\n== Materials suitable for battery_cathode (with rule) ==")
        cathodes = []
        for formula, (nid, _) in sorted(materials.items()):
            for _, dp in g.targets(nid, "suitable_for"):
                if dp["name"] == "battery_cathode":
                    cathodes.append(formula)
        print_sample(cathodes, "materials classified battery_cathode")

        print("\n== Application domain distribution ==")
        by_domain: dict[str, int] = defaultdict(int)
        for _, (nid, _) in materials.items():
            for _, dp in g.targets(nid, "suitable_for"):
                by_domain[dp["name"]] += 1
        for domain, count in sorted(by_domain.items(), key=lambda kv: -kv[1]):
            print(f"  {domain:22s} {count}")

        print("\n== Formula patterns (formula_anonymous groupings) ==")
        by_pattern: dict[str, list[str]] = defaultdict(list)
        no_pattern = 0
        for formula, (nid, _) in sorted(materials.items()):
            targets = g.targets(nid, "has_formula_pattern")
            if targets:
                by_pattern[targets[0][1]["pattern"]].append(formula)
            else:
                no_pattern += 1
        top_patterns = sorted(by_pattern.items(), key=lambda kv: -len(kv[1]))
        print(f"  {len(by_pattern)} distinct patterns across {len(materials)} materials")
        for pattern, formulas in top_patterns[:sample_cap]:
            sample = formulas[:5]
            more = f" ... +{len(formulas) - 5} more" if len(formulas) > 5 else ""
            print(f"    {pattern:10s} ({len(formulas)}): {sample}{more}")
        if no_pattern:
            print(f"  ({no_pattern} materials with no formula_anonymous)")

        print("\n== Oxidation states (possible_species groupings) ==")
        by_species: dict[str, int] = defaultdict(int)
        for _, (nid, _) in materials.items():
            for _, dp in g.targets(nid, "has_oxidation_state"):
                by_species[dp["species"]] += 1
        top_species = sorted(by_species.items(), key=lambda kv: -kv[1])
        print(f"  {len(by_species)} distinct oxidation states across the graph")
        for species, count in top_species[:sample_cap]:
            print(f"    {species:8s} appears in {count} materials")

        print("\n== Elastic / dielectric coverage ==")
        has_elastic = sum(1 for _, p in materials.values() if p.get("bulk_modulus_vrh") is not None)
        has_dielectric = sum(1 for _, p in materials.values() if p.get("dielectric_total") is not None)
        has_species_data = sum(1 for _, (nid, _) in materials.items() if g.targets(nid, "has_oxidation_state"))
        print(f"  bulk_modulus_vrh present:  {has_elastic}/{len(materials)}")
        print(f"  dielectric_total present:  {has_dielectric}/{len(materials)}")
        print(f"  oxidation states present:  {has_species_data}/{len(materials)}")

        print("\n== similar_to edges (sample, sorted by weight) ==")
        seen = set()
        rows = []
        for formula, (nid, _) in materials.items():
            for rel, target_id, props in g.out[nid]:
                if rel != "similar_to":
                    continue
                other_formula = g.node[target_id]["formula"]
                key = tuple(sorted([formula, other_formula]))
                if key in seen:
                    continue
                seen.add(key)
                rows.append((key[0], key[1], props.get("weight", 0.0)))
        rows.sort(key=lambda r: -r[2])
        print_sample(
            [f"{a:14s} <-> {b:14s} weight={weight:.3f}" for a, b, weight in rows],
            "similar_to pairs",
        )

        print("\n== Structural checks ==")
        isolated = [
            formula for formula, (nid, _) in materials.items() if not g.targets(nid, "contains")
        ]
        if isolated:
            print(f"  !! Materials with no element edges: {isolated}")
        else:
            print("  OK: every Material has at least one contains edge.")

        assert len(materials) > 0, "expected at least one Material node"
        print(f"  OK: {len(materials)} Material nodes present.")

        # Each Material index field must be embedded from its own text (guards
        # the cognee 1.3.0 shared-metadata bug patched in matgraph.cognee_patches).
        from cognee.infrastructure.databases.vector import get_vector_engine

        vector_engine = get_vector_engine()
        probe = "LiFePO4" if "LiFePO4" in materials else next(iter(materials))
        top_ids = {}
        for collection in ("Material_formula", "Material_description", "Material_profile"):
            hits = await vector_engine.search(collection, query_text=probe, limit=1, include_payload=True)
            top_ids[collection] = (str(hits[0].id), (hits[0].payload or {}).get("text", "")[:40]) if hits else None
        formula_hit = top_ids["Material_formula"]
        assert formula_hit and g.node[formula_hit[0]]["formula"] == probe, (
            f"Material_formula search for {probe!r} returned {formula_hit} - index fields not embedded separately?"
        )
        payloads = {v[1] for v in top_ids.values() if v}
        assert len(payloads) == 3, f"Material_* collections share embedded text: {top_ids}"
        print(f"  OK: Material_formula / _description / _profile embedded from distinct text ({probe!r} probe).")
