"""Ingestion tests on synthetic MP-shaped records (data/raw is not in git)."""

import asyncio

import pytest

from matgraph.enrich import build_graph, compute_similarity_stats, elastic_status, load_config, load_elements_table
from matgraph.graph_store import KnowledgeBase, graph_from_datapoints
from matgraph.ingest import CompositionSource, IngestError, evaluate_rules, ingest_material


def raw_record(mid, formula, elements, anonymous, species, sg, gap, hull, **extra):
    return {
        "material_id": mid, "formula_pretty": formula, "elements": elements,
        "chemsys": "-".join(sorted(elements)), "formula_anonymous": anonymous, "possible_species": species,
        "symmetry": {"symbol": sg[0], "number": sg[1], "crystal_system": sg[2]},
        "band_gap": gap, "is_gap_direct": False, "is_metal": gap == 0, "formation_energy_per_atom": -2.0,
        "energy_above_hull": hull, "is_stable": hull == 0, "density": 5.0, "volume": 60.0, "nsites": 5,
        "total_magnetization": 0.0, "ordering": "NM", "theoretical": False,
        "description": f"{formula} test structure.", **extra,
    }


CURATED = [
    raw_record("mp-1", "SrTiO3", ["Sr", "Ti", "O"], "ABC3", ["Sr2+", "Ti4+", "O2-"], ("Pm-3m", 221, "Cubic"), 1.8, 0.0),
    raw_record("mp-2", "BaTiO3", ["Ba", "Ti", "O"], "ABC3", ["Ba2+", "Ti4+", "O2-"], ("Pm-3m", 221, "Cubic"), 2.3, 0.0),
    raw_record("mp-3", "Al2O3", ["Al", "O"], "A2B3", ["Al3+", "O2-"], ("R-3c", 167, "Trigonal"), 5.9, 0.0),
    raw_record("mp-4", "GaN", ["Ga", "N"], "AB", ["Ga3+", "N3-"], ("P6_3mc", 186, "Hexagonal"), 1.7, 0.0),
]
NEW = raw_record("mp-9", "LaAlO3", ["La", "Al", "O"], "ABC3", ["La3+", "Al3+", "O2-"], ("Pm-3m", 221, "Cubic"), 4.0, 0.0)


class FixedSource:
    name = "mp"

    def __init__(self, raw):
        self.raw = raw

    def fetch(self, key):
        return dict(self.raw) if key in (self.raw["material_id"], self.raw["formula_pretty"]) else None


@pytest.fixture
def kb():
    graph = graph_from_datapoints(build_graph(CURATED, load_config(), load_elements_table()))
    return KnowledgeBase(graph, None, None)


def ingest(kb, key, sources):
    return asyncio.run(ingest_material(kb, key, sources=sources, curated=CURATED,
                                       stats=compute_similarity_stats(CURATED)))


def test_new_material_reuses_existing_hubs(kb, tmp_path, monkeypatch):
    monkeypatch.setattr("matgraph.ingest.DATA_DIR", tmp_path)
    before = {t: len(ids) for t, ids in kb.graph.by_type.items()}
    report = ingest(kb, "LaAlO3", [FixedSource(NEW)])
    after = {t: len(ids) for t, ids in kb.graph.by_type.items()}

    # Existing hubs reused: O, Al, ABC3, Pm-3m, Cubic, O2-, Al3+ ...
    assert after["FormulaPattern"] == before["FormulaPattern"]
    assert after["SpaceGroup"] == before["SpaceGroup"]
    assert after["CrystalSystem"] == before["CrystalSystem"]
    # ... and only genuinely new ones created: La, La3+, the La-Al-O chemsys.
    assert after["Element"] == before["Element"] + 1
    assert after["OxidationState"] == before["OxidationState"] + 1
    assert after["ChemicalSystem"] == before["ChemicalSystem"] + 1
    assert after["Material"] == before["Material"] + 1

    abc3 = next(h for h in report.shared_hubs if h.node == "FormulaPattern:ABC3")
    assert abc3.shared_with == 2 and not abc3.new_node
    assert any(h.node == "Element:La" and h.new_node for h in report.shared_hubs)
    assert report.similar_materials, "MP record with core properties gets similar_to edges"
    assert {r.domain for r in report.rules if r.status == "fired"} >= {"dielectric"}
    assert not report.persisted


def test_preview_material_is_queryable_as_anchor(kb, tmp_path, monkeypatch):
    monkeypatch.setattr("matgraph.ingest.DATA_DIR", tmp_path)
    ingest(kb, "LaAlO3", [FixedSource(NEW)])
    from matgraph.query_schema import QueryConstraints
    from matgraph.retriever import retrieve

    result = retrieve(kb.graph, QueryConstraints(anchor_materials=["LaAlO3"]))
    assert "LaAlO3" not in [c.formula for c in result.candidates]
    srtio3 = next(c for c in result.candidates if c.formula == "SrTiO3")
    assert "FormulaPattern:ABC3" in srtio3.anchor_links[0].shared_hubs


def test_existing_material_is_not_reingested(kb):
    report = ingest(kb, "SrTiO3", [FixedSource(NEW)])
    assert report.already_in_graph and report.source == "graph"


def test_composition_only_reports_unavailable_not_estimated(kb, tmp_path, monkeypatch):
    monkeypatch.setattr("matgraph.ingest.DATA_DIR", tmp_path)
    report = ingest(kb, "La2Zr2O7", [CompositionSource()])
    assert report.source == "composition"
    props = {p.name: p for p in report.properties}
    assert props["band_gap"].provenance == "unavailable" and props["band_gap"].value is None
    assert props["formula_anonymous"].value == "A2B2C7" and props["formula_anonymous"].provenance == "derived"
    assert props["oxidation_states"].provenance == "guessed"
    assert not report.similar_materials
    assert report.compositional_neighbours
    statuses = {r.domain: r.status for r in report.rules}
    assert statuses["battery_cathode"] == "not_fired"  # no Li: decidable from composition
    assert statuses["dielectric"] == "cannot_evaluate"  # needs band gap


def test_invalid_formula_raises(kb):
    with pytest.raises(IngestError):
        ingest(kb, "NotAFormula!!", [CompositionSource()])


def test_rule_evaluation_partial_for_composition():
    rules = {r.domain: r for r in evaluate_rules({"elements": ["Li", "Fe", "O"]}, load_config())}
    assert rules["battery_cathode"].status == "cannot_evaluate"
    assert rules["photocatalyst"].status == "cannot_evaluate"
    assert rules["dielectric"].status == "cannot_evaluate"


def test_unphysical_elastic_tensor_is_discarded():
    bad = {"bulk_modulus": {"vrh": -300604.4}, "shear_modulus": {"vrh": -180381.6}, "universal_anisotropy": -36903.5}
    ok = {"bulk_modulus": {"vrh": 150.0}, "shear_modulus": {"vrh": 80.0}, "universal_anisotropy": 0.3}
    assert elastic_status(bad) == "unphysical"
    assert elastic_status(ok) == "ok"
    assert elastic_status({}) == "missing"


def test_similarity_zscores_are_clipped():
    from matgraph.enrich import Z_CLIP, similarity_vector

    stats = {f: (0.0, 1.0) for f in ["band_gap", "formation_energy_per_atom", "energy_above_hull", "density",
                                     "total_magnetization", "bulk_modulus_vrh", "shear_modulus_vrh",
                                     "universal_anisotropy", "homogeneous_poisson", "dielectric_total"]}
    raw = {"band_gap": 1.0, "universal_anisotropy": 461.6, "bulk_modulus": {"vrh": 100.0},
           "shear_modulus": {"vrh": 70.0}}
    vec = similarity_vector(raw, stats)
    assert max(abs(v) for v in vec) == Z_CLIP
