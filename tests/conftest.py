"""Shared fixtures: a small synthetic graph in cognee's (nodes, edges) shape.

Tests run without cognee stores, Ollama or OpenRouter: the Graph is built
from hand-written tuples and the LLM is never called (tests use the
rule-based parser / templated explanation, or inject a fake completion).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from matgraph.graph_store import Graph  # noqa: E402

RULES = {
    "battery_cathode": "contains Li and a transition metal, and energy_above_hull <= 0.05 eV/atom",
    "semiconductor_device": "0 < band_gap <= 3.5 eV and not metallic",
    "photocatalyst": "contains O and 1.5 <= band_gap <= 3.5 eV",
    "dielectric": "band_gap > 2 eV and |total_magnetization| < 0.1 (non-magnetic)",
}

# formula: (props, elements, space group (symbol, number, crystal system), classes, domains, species, pattern)
MATERIALS = {
    "LiFePO4": (dict(band_gap=3.7, energy_above_hull=0.0, formation_energy_per_atom=-2.5, density=3.5,
                     total_magnetization=4.0, is_metal=False, is_gap_direct=False, ordering="FM"),
                ["Li", "Fe", "P", "O"], ("Pnma", 62, "Orthorhombic"),
                [("band_gap", "wide_gap"), ("stability", "stable")], ["battery_cathode"],
                ["Li+", "Fe2+", "P5+", "O2-"], "ABCD4"),
    "LiMnPO4": (dict(band_gap=3.9, energy_above_hull=0.0, formation_energy_per_atom=-2.6, density=3.4,
                     total_magnetization=5.0, is_metal=False, is_gap_direct=False, ordering="AFM",
                     bulk_modulus_vrh=95.0),
                ["Li", "Mn", "P", "O"], ("Pnma", 62, "Orthorhombic"),
                [("band_gap", "wide_gap"), ("stability", "stable")], ["battery_cathode"],
                ["Li+", "Mn2+", "P5+", "O2-"], "ABCD4"),
    "LiCoO2": (dict(band_gap=2.3, energy_above_hull=0.0, formation_energy_per_atom=-1.7, density=5.1,
                    total_magnetization=0.0, is_metal=False, is_gap_direct=False, ordering="NM"),
               ["Li", "Co", "O"], ("R-3m", 166, "Trigonal"),
               [("band_gap", "semiconductor"), ("stability", "stable")],
               ["battery_cathode", "semiconductor_device", "photocatalyst", "dielectric"],
               ["Li+", "Co3+", "O2-"], "ABC2"),
    "GaN": (dict(band_gap=1.7, energy_above_hull=0.0, formation_energy_per_atom=-0.7, density=6.1,
                 total_magnetization=0.0, is_metal=False, is_gap_direct=True, ordering="NM",
                 bulk_modulus_vrh=172.0),
            ["Ga", "N"], ("P6_3mc", 186, "Hexagonal"),
            [("band_gap", "semiconductor"), ("stability", "stable")], ["semiconductor_device"],
            ["Ga3+", "N3-"], "AB"),
    "BaTiO3": (dict(band_gap=2.3, energy_above_hull=0.0, formation_energy_per_atom=-3.5, density=5.9,
                    total_magnetization=0.0, is_metal=False, is_gap_direct=False, ordering="NM",
                    bulk_modulus_vrh=98.0, dielectric_total=13.0),
               ["Ba", "Ti", "O"], ("Amm2", 38, "Orthorhombic"),
               [("band_gap", "semiconductor"), ("stability", "stable")],
               ["semiconductor_device", "photocatalyst", "dielectric"], ["Ba2+", "Ti4+", "O2-"], "ABC3"),
    "Fe3O4": (dict(band_gap=0.0, energy_above_hull=0.03, formation_energy_per_atom=-1.6, density=5.2,
                   total_magnetization=16.0, is_metal=True, is_gap_direct=False, ordering="FiM"),
              ["Fe", "O"], ("Fd-3m", 227, "Cubic"),
              [("band_gap", "metal"), ("stability", "metastable")], [], ["Fe2+", "Fe3+", "O2-"], "A3B4"),
}
SIMILAR = [("LiFePO4", "LiMnPO4", 0.95), ("LiMnPO4", "LiFePO4", 0.95), ("LiCoO2", "BaTiO3", 0.7)]


def build_test_graph() -> Graph:
    nodes: list[tuple[str, dict]] = []
    edges: list[tuple[str, str, str, dict]] = []
    ids: dict[tuple, str] = {}

    def node(key: tuple, props: dict) -> str:
        if key not in ids:
            ids[key] = f"n{len(ids)}"
            nodes.append((ids[key], props))
        return ids[key]

    for formula, (props, elements, sg, classes, domains, species, pattern) in MATERIALS.items():
        mid = node(("Material", formula), dict(type="Material", formula=formula,
                                               material_id=f"mp-{formula.lower()}", **props))
        for sym in elements:
            edges.append((mid, node(("Element", sym), dict(type="Element", symbol=sym)), "contains", {}))
        cs = node(("CrystalSystem", sg[2]), dict(type="CrystalSystem", name=sg[2]))
        sg_key = ("SpaceGroup", sg[0])
        new_sg = sg_key not in ids
        sgid = node(sg_key, dict(type="SpaceGroup", symbol=sg[0], number=sg[1]))
        if new_sg:
            edges.append((sgid, cs, "crystal_system", {}))
        edges.append((mid, sgid, "has_space_group", {}))
        for kind, name in classes:
            pc = node(("PropertyClass", kind, name), dict(type="PropertyClass", kind=kind, name=name))
            edges.append((mid, pc, "classified_as", {}))
        for d in domains:
            dn = node(("ApplicationDomain", d), dict(type="ApplicationDomain", name=d))
            edges.append((mid, dn, "suitable_for", {"properties": {"rule": RULES[d]}}))
        for sp in species:
            edges.append((mid, node(("OxidationState", sp), dict(type="OxidationState", species=sp)),
                          "has_oxidation_state", {}))
        edges.append((mid, node(("FormulaPattern", pattern), dict(type="FormulaPattern", pattern=pattern)),
                      "has_formula_pattern", {}))
    for a, b, w in SIMILAR:
        edges.append((ids[("Material", a)], ids[("Material", b)], "similar_to", {"weight": w}))
    return Graph(nodes, edges)


@pytest.fixture
def graph() -> Graph:
    return build_test_graph()


@pytest.fixture
def retrieval_config() -> dict:
    return {
        "missing_value_policy": "exclude",
        "ranking_weights": {"constraint_margin": 0.35, "anchor": 0.30, "semantic": 0.25, "stability": 0.10},
        "anchor": {"hub_weight": 0.5, "similarity_weight": 0.5, "two_hop_discount": 0.8,
                   "hub_relations": ["contains", "has_oxidation_state", "has_formula_pattern",
                                     "has_space_group", "member_of", "classified_as", "suitable_for"]},
        "stability_scale": 0.025,
        "semantic_collections": ["Material_profile", "Material_description"],
    }
