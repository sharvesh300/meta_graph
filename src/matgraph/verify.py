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

import json
import logging
from collections import defaultdict

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.users.methods import get_default_user

from matgraph.enrich import DATA_DIR
from matgraph.pipeline import DATASET_NAME

logger = logging.getLogger(__name__)


class Graph:
    """Thin in-memory index over cognee's (nodes, edges) tuples."""

    def __init__(self, nodes, edges):
        self.node = {node_id: props for node_id, props in nodes}
        self.out: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
        self.in_: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
        for source_id, target_id, rel, props in edges:
            self.out[source_id].append((rel, target_id, props))
            self.in_[target_id].append((rel, source_id, props))

    def nodes_of_type(self, type_name: str):
        return [(nid, p) for nid, p in self.node.items() if p.get("type") == type_name]

    def targets(self, node_id: str, rel: str):
        """(target_id, target_node_props) pairs for outgoing edges of type `rel`."""
        return [(tid, self.node[tid]) for r, tid, _edge_props in self.out[node_id] if r == rel]

    def target_edges(self, node_id: str, rel: str):
        """(target_id, edge_props) pairs, when the edge's own properties matter."""
        return [(tid, ep) for r, tid, ep in self.out[node_id] if r == rel]


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
        problems: list[str] = []

        # Counted off the node list rather than the `materials` dict above:
        # that dict is keyed by formula, so any duplicate formula would collapse
        # two nodes into one entry and undercount.
        material_nodes = g.nodes_of_type("Material")

        def check(passed: bool, ok_message: str, problem: str) -> None:
            """Report a check, and remember it if it failed."""
            if passed:
                print(f"  OK: {ok_message}")
            else:
                print(f"  !! {problem}")
                problems.append(problem)

        def sample(formulas: list[str], cap: int = 10) -> str:
            shown = ", ".join(formulas[:cap])
            return shown + (f", ... (+{len(formulas) - cap} more)" if len(formulas) > cap else "")

        check(
            bool(material_nodes),
            f"{len(material_nodes)} Material nodes present.",
            "no Material nodes in the graph",
        )

        isolated = [p["formula"] for nid, p in material_nodes if not g.targets(nid, "contains")]
        check(
            not isolated,
            "every Material has at least one contains edge (no isolated materials).",
            f"{len(isolated)} Material(s) with no contains edge: {sample(isolated)}",
        )

        # Every Material should come from exactly one cached record, so the node
        # count must match the fetch manifest. A mismatch means either records
        # were skipped as incomplete (enrich.filter_valid_materials logs which)
        # or two materials collided on their material_id identity field.
        manifest_path = DATA_DIR / "_manifest.json"
        if manifest_path.exists():
            expected = len(json.loads(manifest_path.read_text()))
            check(
                len(material_nodes) == expected,
                f"Material count matches the fetch manifest exactly ({expected}).",
                f"graph has {len(material_nodes)} Material nodes but the manifest "
                f"lists {expected} cached records",
            )
        else:
            print(f"  -- manifest check skipped: {manifest_path} not found")

        no_space_group = [
            p["formula"] for nid, p in material_nodes if len(g.targets(nid, "has_space_group")) != 1
        ]
        check(
            not no_space_group,
            "every Material has exactly one space group.",
            f"{len(no_space_group)} Material(s) without exactly one has_space_group "
            f"edge: {sample(no_space_group)}",
        )

        unclassified = [
            p["formula"] for nid, p in material_nodes if not g.targets(nid, "classified_as")
        ]
        check(
            not unclassified,
            "every Material has at least one property classification.",
            f"{len(unclassified)} Material(s) with no classified_as edge: {sample(unclassified)}",
        )

        # Informational, not a failure: enrich.py deliberately skips a single
        # classification whose value is missing or unbucketable rather than
        # failing the build, and logs a warning when it does. This surfaces the
        # after-effect in the built graph.
        for kind in ("band_gap", "stability"):
            missing = [
                p["formula"]
                for nid, p in material_nodes
                if kind not in {cp["kind"] for _, cp in g.targets(nid, "classified_as")}
            ]
            if missing:
                print(
                    f"  -- {len(missing)} Material(s) have no '{kind}' classification "
                    f"(see enrich.py warnings from the build): {sample(missing)}"
                )

        if problems:
            raise AssertionError(
                f"graph verification failed with {len(problems)} problem(s):\n  - "
                + "\n  - ".join(problems)
            )
        print("\n  All structural checks passed.")
