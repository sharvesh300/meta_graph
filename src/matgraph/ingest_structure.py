"""Structure ingestion (docs/PHASE3_PLAN.md §5): a user-supplied crystal
structure -> CGCNN predictions -> a ``struct-`` Material related to the graph.

1. Check it (``cgcnn.validate``: size, sites, ordered, elements, distances).
2. Deduplicate: StructureMatcher against every graph material with the same
   reduced formula (MP records from data/raw/, persisted structures from
   data/raw/user_structures/). A match returns the existing node - for an MP
   material together with the CGCNN prediction next to its DFT values.
   A new polymorph of a formula that is already present is a new node.
3. Build an MP-summary-shaped record: facts derived exactly from the
   structure (formula, chemsys, pattern, space group, volume, density, nsites;
   oxidation states guessed by bond valence) plus ``cgcnn.interpret`` values.
   DFT fields stay empty; predictions go to ``predicted_*``.
4. Predicted edges, all marked ``provenance: predicted``:
   - band-gap class: "metal" when the classifier says so; otherwise the
     bucket of the predicted gap, only if the gap is more than its expected
     error away from every bucket boundary;
   - application rules that hold across gap ± error (ingest's predicted
     rule evaluation). No stability class: CGCNN does not predict E_hull.
5. Relate, report, preview or persist through ``ingest.relate_and_report``;
   persisting also stores the record (structure + predictions) so a rebuild
   replays it without re-running the model.
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any

from pymatgen.core import Composition, Structure

from matgraph.cgcnn.validate import SYMPREC, StructureError, allowed_elements, check_structure, struct_id
from matgraph.enrich import DATA_DIR, NodeFactory, classify_band_gap, load_config, load_elements_table
from matgraph.graph_store import KnowledgeBase
from matgraph.ingest import (
    IngestError,
    RelationReport,
    _species_string,
    evaluate_rules,
    predicted_properties,
    relate_and_report,
    relation_report_for_existing,
)
from matgraph.schema import Edge, Material

logger = logging.getLogger(__name__)

USER_STRUCTURES_DIR = DATA_DIR / "user_structures"
# Structures of preview-only (overlay) struct nodes, so a repeated preview in
# the same session still deduplicates.
_preview_structures: dict[str, Structure] = {}


@functools.cache
def get_predictor():
    from matgraph.cgcnn.predictor import CGCNNPredictor

    return CGCNNPredictor()


# --- record -----------------------------------------------------------------------


def _guess_species(structure: Structure) -> list[str]:
    from pymatgen.analysis.bond_valence import BVAnalyzer

    try:
        valences = BVAnalyzer().get_valences(structure)
        pairs = {(site.specie.symbol, v) for site, v in zip(structure, valences)}
    except Exception:  # noqa: BLE001 - BVAnalyzer raises ValueError on failure, others on odd inputs
        try:
            guesses = structure.composition.oxi_state_guesses() if structure.composition.num_atoms <= 60 else []
        except Exception:  # noqa: BLE001
            guesses = []
        pairs = set(guesses[0].items()) if guesses else set()
    return sorted({s for sym, v in pairs if (s := _species_string(sym, v))})


def derive_record(structure: Structure, material_id: str) -> dict[str, Any]:
    """Exact facts of the structure, in MP summary field names."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    sga = SpacegroupAnalyzer(structure, symprec=SYMPREC)
    comp: Composition = structure.composition
    elements = sorted(el.symbol for el in comp.elements)
    symbol, number = sga.get_space_group_symbol(), sga.get_space_group_number()
    return {
        "material_id": material_id,
        "formula_pretty": comp.reduced_formula,
        "elements": elements,
        "chemsys": "-".join(elements),
        "formula_anonymous": comp.anonymized_formula,
        "possible_species": _guess_species(structure),
        "_species_guessed": True,
        "symmetry": {"symbol": symbol, "number": number, "crystal_system": sga.get_crystal_system().capitalize()},
        "nsites": len(structure),
        "volume": float(structure.volume),
        "density": float(structure.density),
        "structure": structure.as_dict(),
        "description": f"{comp.reduced_formula}: user-supplied crystal structure, {len(structure)} sites, "
                       f"space group {symbol} (#{number}). No Materials Project record.",
    }


def predict_record(structure: Structure) -> tuple[dict[str, Any], dict[str, Any]]:
    """(predicted_* attributes, full interpreted predictions for storage)."""
    from matgraph.cgcnn.interpret import interpret
    from matgraph.cgcnn.registry import MODEL_SET

    values = interpret(get_predictor().predict(structure))
    attrs: dict[str, Any] = {name: v.value for name, v in values.items()}
    attrs["predicted_metal_probability"] = values["predicted_is_metal"].detail["p_metal"]
    attrs["prediction_model"] = f"cgcnn-{MODEL_SET}"
    stored = {name: {"value": v.value, "units": v.units, "model_ids": list(v.model_ids),
                     "expected_error": v.expected_error, "expected_error_units": v.expected_error_units,
                     "detail": v.detail, "flags": list(v.flags)} for name, v in values.items()}
    return attrs, stored


# --- deduplication ----------------------------------------------------------------------


def _stored_structure(material_id: str) -> Structure | None:
    if material_id in _preview_structures:
        return _preview_structures[material_id]
    for path in (DATA_DIR / f"{material_id}.json", USER_STRUCTURES_DIR / f"{material_id}.json"):
        if path.exists():
            data = json.loads(path.read_text()).get("structure")
            return Structure.from_dict(data) if data else None
    return None


def find_matching_material(kb: KnowledgeBase, structure: Structure) -> str | None:
    """Node id of a graph material with the same formula and a matching structure."""
    from pymatgen.analysis.structure_matcher import StructureMatcher

    formula = structure.composition.reduced_formula
    matcher = StructureMatcher()
    for node_id, props in kb.graph.nodes_of_type("Material"):
        if props.get("formula") != formula:
            continue
        other = _stored_structure(props["material_id"])
        if other is not None and matcher.fit(structure, other):
            return node_id
    return None


# --- predicted edges and profile ---------------------------------------------------------


def _hub_description(kb: KnowledgeBase, kind: str, name: str, fallback: str) -> str:
    """Reuse the existing PropertyClass description, so upserting this
    material does not rewrite the shared hub with a predicted value."""
    node_id = kb.graph.find("PropertyClass", kind=kind, name=name)
    return kb.graph.node[node_id]["description"] if node_id else fallback


def predicted_band_gap_class(raw: dict[str, Any], config: dict[str, Any]) -> str | None:
    from matgraph.cgcnn.interpret import expected_errors

    if raw["predicted_is_metal"]:
        return "metal"
    gap = raw["predicted_band_gap"]
    err = (expected_errors().get("predicted_band_gap") or {}).get("value") or 0.0
    boundaries = {b[k] for b in config["band_gap_buckets"] for k in ("min", "max") if k in b}
    if any(abs(gap - b) <= err for b in boundaries):
        return None
    return classify_band_gap(gap, config)["name"]


def _fmt_error(stored: dict[str, Any]) -> str:
    err, units = stored.get("expected_error"), stored.get("expected_error_units") or ""
    if err is None:
        return ""
    if units.startswith("log10"):
        return f" (typically within a factor of {10 ** err:.2f})"
    return f" (expected error ±{err:.2g}{(' ' + units) if units and units != 'error rate' else ''})"


def build_profile(raw: dict[str, Any], gap_class: str | None, domains: list[str],
                  elements_table: dict[str, Any]) -> str:
    p = raw["_predicted"]
    sym = raw["symmetry"]
    names = ", ".join(elements_table[s]["name"] for s in raw["elements"])
    metal = p["predicted_is_metal"]
    parts = [
        f"{raw['formula_pretty']} ({raw['material_id']}) is a user-supplied crystal structure with no Materials "
        f"Project record, in the {raw['chemsys']} chemical system, space group {sym['symbol']} "
        f"({sym['crystal_system'].lower()} crystal system), {raw['nsites']} sites, density {raw['density']:.2f} g/cm3.",
        f"Contains {names}.",
    ]
    if raw.get("possible_species"):
        parts.append(f"Oxidation states (guessed): {', '.join(raw['possible_species'])}.")
    parts.append(f"Formula pattern {raw['formula_anonymous']}.")
    preds = [
        f"formation energy ≈ {p['predicted_formation_energy_per_atom']['value']:.3f} eV/atom"
        + _fmt_error(p["predicted_formation_energy_per_atom"]),
        ("metallic" if metal["value"] else "non-metallic")
        + f" (P(metal) = {metal['detail']['p_metal']:.2f}){_fmt_error(metal)}",
        f"band gap ≈ {p['predicted_band_gap']['value']:.2f} eV" + _fmt_error(p["predicted_band_gap"])
        + (f", {gap_class}" if gap_class else ""),
        f"Fermi energy ≈ {p['predicted_efermi']['value']:.2f} eV" + _fmt_error(p["predicted_efermi"]),
        f"bulk modulus ≈ {p['predicted_bulk_modulus_vrh']['value']:.0f} GPa" + _fmt_error(p["predicted_bulk_modulus_vrh"]),
        f"shear modulus ≈ {p['predicted_shear_modulus_vrh']['value']:.0f} GPa" + _fmt_error(p["predicted_shear_modulus_vrh"]),
        f"Poisson ratio ≈ {p['predicted_homogeneous_poisson']['value']:.2f}" + _fmt_error(p["predicted_homogeneous_poisson"]),
    ]
    parts.append("CGCNN predictions (pre-trained models, not DFT values): " + "; ".join(preds) + ".")
    if domains:
        parts.append(f"Predicted suitable for: {', '.join(d.replace('_', ' ') for d in domains)}.")
    parts.append("Not available (no DFT calculation): stability / energy above hull, magnetization, "
                 "dielectric properties.")
    return " ".join(parts)


def apply_predictions(kb: KnowledgeBase, material: Material, raw: dict[str, Any], factory: NodeFactory,
                      config: dict[str, Any], elements_table: dict[str, Any]) -> list[str]:
    """Set predicted attributes, predicted edges and the profile; returns notes."""
    for name, value in raw.items():
        if name.startswith("predicted_") or name == "prediction_model":
            setattr(material, name, value)
    material.property_source = "cgcnn"
    model = raw["prediction_model"]
    marks = {"provenance": "predicted", "model": model}
    notes = []

    gap_class = predicted_band_gap_class(raw, config)
    if gap_class:
        description = _hub_description(kb, "band_gap", gap_class, f"band_gap classified as '{gap_class}'")
        material.classified_as = [(Edge(relationship_type="classified_as", properties=dict(marks)),
                                   factory.property_class("band_gap", gap_class, description))]
    else:
        notes.append(f"No band-gap class: the predicted gap {raw['predicted_band_gap']:.2f} eV is within its "
                     "expected error of a class boundary.")

    fired = [r for r in evaluate_rules(raw, config) if r.status == "fired"]
    material.suitable_for = [
        (Edge(relationship_type="suitable_for", properties={"rule": r.rule, **marks}), factory.domain(r.domain))
        for r in fired
    ]
    material.profile = build_profile(raw, gap_class, [r.domain for r in fired], elements_table)
    return notes


# --- entry point ------------------------------------------------------------------------------


def _comparison_note(report: RelationReport, raw_predicted: dict[str, Any]) -> str:
    dft = {p.name: p.value for p in report.properties if p.provenance == "mp"}
    pairs = [("formation_energy_per_atom", "predicted_formation_energy_per_atom", "eV/atom", 3),
             ("band_gap", "predicted_band_gap", "eV", 2), ("efermi", "predicted_efermi", "eV", 2),
             ("bulk_modulus_vrh", "predicted_bulk_modulus_vrh", "GPa", 0),
             ("shear_modulus_vrh", "predicted_shear_modulus_vrh", "GPa", 0)]
    shown = [f"{d} {raw_predicted[p]:.{n}f} vs {dft[d]:.{n}f} {u}" for d, p, u, n in pairs
             if dft.get(d) is not None and raw_predicted.get(p) is not None]
    return ("CGCNN on the uploaded structure vs Materials Project DFT (predicted vs DFT): " + "; ".join(shown) + "."
            if shown else "")


async def ingest_structure(
    kb: KnowledgeBase,
    structure: Structure | None = None,
    label: str = "",
    persist: bool = False,
    record: dict[str, Any] | None = None,
) -> RelationReport:
    """Ingest a structure (or replay a stored record) as a struct- Material."""
    config, elements_table = load_config(), load_elements_table()
    g = kb.graph

    if record is not None:  # replay: stored structure and predictions, no model run
        raw = dict(record)
        existing = g.material(raw["material_id"])
        if existing is not None:
            return relation_report_for_existing(kb, existing, label or raw["material_id"])
        notes = [f"Replayed {raw['material_id']} from its stored structure and predictions."]
    else:
        if structure is None:
            raise IngestError("no structure given")
        try:
            check_structure(structure, allowed_elements(elements_table), set(config["exclude_elements"]))
        except StructureError as error:
            raise IngestError(str(error)) from error
        label = label or structure.composition.reduced_formula

        match = find_matching_material(kb, structure)
        if match is not None:
            report = relation_report_for_existing(kb, match, label)
            if not g.node[match]["material_id"].startswith("struct-"):
                attrs, _ = predict_record(structure)
                report.properties += predicted_properties({**attrs})
                if note := _comparison_note(report, attrs):
                    report.notes.append(note)
            report.notes.insert(0, f"The uploaded structure matches {g.node[match]['formula']} "
                                   f"({g.node[match]['material_id']}) already in the graph (StructureMatcher).")
            return report

        material_id = struct_id(structure)
        base, n = material_id, 2
        while g.material(material_id) is not None:  # same name, but StructureMatcher says a different crystal
            material_id, n = f"{base}-{n}", n + 1
        raw = derive_record(structure, material_id)
        attrs, stored = predict_record(structure)
        raw.update(attrs)
        raw["_predicted"] = stored
        notes = [f"New crystal structure named {material_id}: properties are CGCNN predictions "
                 f"({attrs['prediction_model']}), not DFT values; no Materials Project record."]
        flags = sorted({f for v in stored.values() for f in v["flags"]})
        if flags:
            notes.append(f"Prediction flags: {', '.join(flags)}.")

    factory = NodeFactory(config, elements_table)
    material = factory.material(raw, origin="user")
    notes += apply_predictions(kb, material, raw, factory, config, elements_table)
    notes.append("No similar_to edges: they compare DFT properties, which a predicted structure does not have; "
                 "related by composition here, and by structurally_similar edges from Phase 3 M5.")

    if persist:
        USER_STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
        path = USER_STRUCTURES_DIR / f"{raw['material_id']}.json"
        if not path.exists():
            path.write_text(json.dumps(raw, indent=2))
    else:
        _preview_structures[raw["material_id"]] = Structure.from_dict(raw["structure"])
    # similar_to is skipped (relate_and_report needs 5 DFT dimensions; a struct record has at most density).
    return await relate_and_report(kb, label or raw["material_id"], raw, "cgcnn", material, factory, config,
                                   notes, persist)
