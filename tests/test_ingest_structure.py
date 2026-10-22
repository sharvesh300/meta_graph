"""Structure ingestion (Phase 3 M2) on a synthetic graph, with a fake CGCNN
predictor: validation, struct- naming, dedup, derived and predicted edges,
persistence and replay. No torch, cognee stores or network."""

import asyncio
import json
import re
from types import SimpleNamespace

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import matgraph.ingest_structure as ist
from matgraph.cgcnn.registry import MODELS
from matgraph.cgcnn.validate import (
    MAX_INPUT_BYTES, StructureError, allowed_elements, check_structure, parse_structure, struct_id,
)
from matgraph.enrich import build_graph, load_config, load_elements_table
from matgraph.graph_store import Graph, KnowledgeBase, graph_from_datapoints
from matgraph.ingest import IngestError, RelationReport, _predicted_rule_evaluations, replay_user_materials


def rocksalt_mgo() -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.21), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def wurtzite(a: str, b: str) -> Structure:
    return Structure.from_spacegroup("P6_3mc", Lattice.hexagonal(3.19, 5.19), [a, b],
                                     [[1 / 3, 2 / 3, 0], [1 / 3, 2 / 3, 0.375]])


def mp_record(mid, structure, gap, **extra):
    sga = SpacegroupAnalyzer(structure, symprec=0.1)
    comp = structure.composition
    elements = sorted(el.symbol for el in comp.elements)
    return {
        "material_id": mid, "formula_pretty": comp.reduced_formula, "elements": elements,
        "chemsys": "-".join(elements), "formula_anonymous": comp.anonymized_formula, "possible_species": [],
        "symmetry": {"symbol": sga.get_space_group_symbol(), "number": sga.get_space_group_number(),
                     "crystal_system": sga.get_crystal_system().capitalize()},
        "band_gap": gap, "is_gap_direct": True, "is_metal": gap == 0, "formation_energy_per_atom": -3.0,
        "energy_above_hull": 0.0, "is_stable": True, "density": float(structure.density),
        "volume": float(structure.volume), "nsites": len(structure), "total_magnetization": 0.0, "ordering": "NM",
        "theoretical": False, "description": f"{comp.reduced_formula} test structure.", **extra,
    }


STRUCTURES = {"mp-10": rocksalt_mgo(), "mp-11": wurtzite("Ga", "N")}
CURATED = [mp_record("mp-10", STRUCTURES["mp-10"], 4.5), mp_record("mp-11", STRUCTURES["mp-11"], 1.7)]


def fake_predictor(gap=2.0, p_metal=0.05):
    outputs = {"formation-energy-per-atom": -2.9, "band-gap": gap, "semi-metal-classification": p_metal,
               "efermi": 1.0, "bulk-moduli": 2.2, "shear-moduli": 1.9, "poisson-ratio": 0.25}

    class Fake:
        def predict(self, structure):
            return {spec.name: SimpleNamespace(value=outputs[spec.name], flags=(), model_id=spec.model_id)
                    for spec in MODELS}

    return Fake()


@pytest.fixture
def kb(tmp_path, monkeypatch):
    monkeypatch.setattr(ist, "USER_STRUCTURES_DIR", tmp_path / "user_structures")
    monkeypatch.setattr("matgraph.ingest.USER_MATERIALS_PATH", tmp_path / "_user_materials.json")
    monkeypatch.setattr(ist, "_stored_structure",
                        lambda mid: STRUCTURES.get(mid) or ist._preview_structures.get(mid) or _persisted(tmp_path, mid))
    monkeypatch.setattr(ist, "get_predictor", lambda: fake_predictor())
    ist._preview_structures.clear()
    return KnowledgeBase(graph_from_datapoints(build_graph(CURATED, load_config(), load_elements_table())), None, None)


def _persisted(tmp_path, mid):
    path = tmp_path / "user_structures" / f"{mid}.json"
    return Structure.from_dict(json.loads(path.read_text())["structure"]) if path.exists() else None


def ingest(kb, structure, **kwargs) -> RelationReport:
    return asyncio.run(ist.ingest_structure(kb, structure, **kwargs))


# --- validation and naming ---------------------------------------------------------------


def allowed():
    return allowed_elements(load_elements_table()), set(load_config()["exclude_elements"])


@pytest.mark.parametrize("make, message", [
    (lambda: rocksalt_mgo() * (4, 4, 4), "limit is 200"),
    (lambda: Structure(Lattice.cubic(4.0), [{"Fe": 0.5, "Ni": 0.5}], [[0, 0, 0]]), "partial occupancies"),
    (lambda: Structure(Lattice.cubic(4.0), ["U", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]), "excluded elements: U"),
    (lambda: Structure(Lattice.cubic(4.0), ["Mg", "O"], [[0, 0, 0], [0.05, 0, 0]]), "closer than 0.5"),
])
def test_check_structure_rejects(make, message):
    with pytest.raises(StructureError, match=message):
        check_structure(make(), *allowed())


def test_parse_limits_and_errors():
    with pytest.raises(StructureError, match="larger than"):
        parse_structure("x" * (MAX_INPUT_BYTES + 1))
    with pytest.raises(StructureError, match="could not parse"):
        parse_structure("not a structure")
    s = parse_structure(rocksalt_mgo().to(fmt="cif"))
    assert s.composition.reduced_formula == "MgO"


def test_struct_id_format_and_invariance():
    s = wurtzite("Mg", "O")
    sid = struct_id(s)
    assert re.fullmatch(r"struct-MgO-sg186-[0-9a-f]{8}", sid)
    shuffled = Structure.from_sites(list(reversed(list(s))))
    shuffled.translate_sites(list(range(len(shuffled))), [0.1, 0.2, 0.3])
    assert struct_id(s * (2, 1, 1)) == struct_id(shuffled) == sid
    assert struct_id(rocksalt_mgo()) != sid


# --- ingestion ---------------------------------------------------------------------------------


def test_new_polymorph_becomes_struct_node(kb):
    report = ingest(kb, wurtzite("Mg", "O"), label="MgO_w.cif")
    g = kb.graph
    assert report.source == "cgcnn" and not report.already_in_graph and not report.persisted
    assert re.fullmatch(r"struct-MgO-sg186-[0-9a-f]{8}", report.material_id)
    props = {p.name: p for p in report.properties}
    assert props["density"].provenance == "derived" and props["space_group"].provenance == "derived"
    assert props["band_gap"].provenance == "unavailable"  # DFT field stays empty
    assert props["predicted_band_gap"].provenance == "predicted"
    assert props["predicted_band_gap"].value == pytest.approx(2.0)
    assert props["predicted_band_gap"].expected_error > 0
    assert props["predicted_bulk_modulus_vrh"].value == pytest.approx(10 ** 2.2)

    # A formula still resolves to the Materials Project node, not the polymorph.
    assert g.node[g.material("MgO")]["material_id"] == "mp-10"
    node = g.node[g.material(report.material_id)]
    assert node["band_gap"] is None and node["property_source"] == "cgcnn"
    assert node["predicted_band_gap"] == pytest.approx(2.0)

    # Shares the wurtzite space group with GaN; band-gap class rests on the prediction.
    sg = next(h for h in report.shared_hubs if h.relation == "has_space_group")
    assert sg.node.startswith("SpaceGroup:P6_3mc") and sg.shared_with == 1 and not sg.predicted
    cls = next(h for h in report.shared_hubs if h.relation == "classified_as")
    assert cls.node == "PropertyClass:band_gap/semiconductor" and cls.predicted
    assert {r.domain for r in report.rules if r.predicted} == {"semiconductor_device", "photocatalyst"}
    assert not report.similar_materials
    assert report.compositional_neighbours[0].material_id == "mp-10"


def test_same_crystal_in_another_setting_resolves_to_mp_node(kb):
    primitive = SpacegroupAnalyzer(rocksalt_mgo()).get_primitive_standard_structure()
    report = ingest(kb, primitive)
    assert report.already_in_graph and report.material_id == "mp-10"
    assert any(p.name == "predicted_band_gap" and p.provenance == "predicted" for p in report.properties)
    assert any(n.startswith("CGCNN on the uploaded structure vs Materials Project DFT") for n in report.notes)


def test_repeated_preview_dedups_onto_overlay_node(kb):
    first = ingest(kb, wurtzite("Mg", "O"))
    second = ingest(kb, wurtzite("Mg", "O") * (1, 1, 2))
    assert second.already_in_graph and second.material_id == first.material_id


def test_gap_near_class_boundary_gets_no_class(kb, monkeypatch):
    monkeypatch.setattr(ist, "get_predictor", lambda: fake_predictor(gap=1.2))
    report = ingest(kb, wurtzite("Mg", "O"))
    assert not any(h.relation == "classified_as" for h in report.shared_hubs)
    assert any("within its expected error of a class boundary" in n for n in report.notes)


def test_predicted_metal_gates_gap(kb, monkeypatch):
    monkeypatch.setattr(ist, "get_predictor", lambda: fake_predictor(gap=2.5, p_metal=0.6))
    report = ingest(kb, wurtzite("Mg", "O"))
    props = {p.name: p.value for p in report.properties}
    assert props["predicted_is_metal"] is True and props["predicted_band_gap"] == 0.0
    assert any(h.node == "PropertyClass:band_gap/metal" and h.predicted for h in report.shared_hubs)
    semi = next(r for r in report.rules if r.domain == "semiconductor_device")
    assert semi.status == "not_fired" and "predicted metallic" in semi.reason


def test_rejected_structure_is_an_ingest_error(kb):
    with pytest.raises(IngestError, match="partial occupancies"):
        ingest(kb, Structure(Lattice.cubic(4.0), [{"Mg": 0.5, "Ca": 0.5}, "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]))


@pytest.mark.parametrize("gap, expected", [
    (2.5, {"semiconductor_device": "fired", "photocatalyst": "fired", "dielectric": "cannot_evaluate"}),
    (3.3, {"semiconductor_device": "cannot_evaluate", "photocatalyst": "cannot_evaluate"}),
    (0.8, {"photocatalyst": "not_fired", "dielectric": "not_fired"}),  # 0.8 + 0.62 < 1.5
])
def test_predicted_rules_respect_expected_error(gap, expected):
    config = load_config()
    rules = {r["domain"]: r["rule"] for r in config["application_rules"]}
    raw = {"elements": ["Mg", "O"], "predicted_band_gap": gap, "predicted_is_metal": False,
           "predicted_metal_probability": 0.05}
    status = {r.domain: r.status for r in _predicted_rule_evaluations(raw, config, rules)}
    assert {d: status[d] for d in expected} == expected
    assert status["battery_cathode"] == "not_fired" and status["magnetic_material"] == "cannot_evaluate"


# --- persistence and replay ----------------------------------------------------------------------


def test_persist_then_replay_reproduces_node(kb, tmp_path, monkeypatch):
    written = []

    async def fake_add_to_graph(datapoints):
        written.extend(datapoints)

    async def fake_load_graph():
        return graph_from_datapoints(build_graph(CURATED, load_config(), load_elements_table()) + written)

    monkeypatch.setattr("matgraph.pipeline.add_to_graph", fake_add_to_graph)
    monkeypatch.setattr("matgraph.graph_store.load_graph", fake_load_graph)

    report = ingest(kb, wurtzite("Mg", "O"), label="MgO_w.cif", persist=True)
    assert report.persisted
    record = json.loads((tmp_path / "user_structures" / f"{report.material_id}.json").read_text())
    assert record["structure"] and record["_predicted"]["predicted_band_gap"]["value"] == pytest.approx(2.0)
    entries = json.loads((tmp_path / "_user_materials.json").read_text())
    assert entries == [{"material_id": report.material_id, "formula": "MgO", "source": "cgcnn", "input": "MgO_w.cif"}]
    persisted = _node_summary(kb.graph, report.material_id)

    # Replay into a fresh graph (as after a rebuild): from the stored record, no model run.
    monkeypatch.setattr(ist, "get_predictor", lambda: pytest.fail("replay must not re-run the model"))
    written.clear()
    fresh = KnowledgeBase(graph_from_datapoints(build_graph(CURATED, load_config(), load_elements_table())), None, None)
    assert asyncio.run(replay_user_materials(fresh)) == ["MgO"]
    assert _node_summary(fresh.graph, report.material_id) == persisted


def _node_summary(g: Graph, material_id: str):
    nid = g.material(material_id)
    props = {k: v for k, v in g.node[nid].items() if k not in ("created_at", "updated_at")}
    return props, sorted((rel, g.node[t].get("type"), json.dumps(ep, sort_keys=True, default=str))
                               for rel, t, ep in g.out[nid])
