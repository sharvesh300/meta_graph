"""New-material ingestion (docs/PHASE2_PLAN.md §6): get a material's
properties and relate it to the existing graph.

Inputs here are a Materials Project id or a formula; crystal structures go
through ``matgraph.ingest_structure`` (Phase 3), which shares the relate /
report / persist core below.

Properties come from a pluggable **property source**:
- ``MPSource``: full summary record from Materials Project (provenance "mp").
- ``CompositionSource``: formula Materials Project has no entry for; only
  what the formula itself determines (provenance "derived"/"guessed"); every
  computed property is reported unavailable, never estimated.
- ``ingest_structure.CGCNNSource``: a user-supplied structure; exact facts
  derived from it ("derived") plus CGCNN predictions ("predicted"), which
  never fill the DFT fields.

The material is built with the same ``enrich.NodeFactory`` as the full graph,
so identity-keyed hub nodes (Element, SpaceGroup, OxidationState, ...) get the
ids of the existing ones and the new material attaches to them. By default
it is only added to the in-memory overlay (preview); ``persist=True`` writes
it to cognee.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from matgraph.enrich import (
    DATA_DIR,
    NodeFactory,
    apply_application_rules,
    count_similarity_dimensions,
    load_config,
    load_elements_table,
    load_raw_materials,
    load_similarity_stats,
    similarity_vector,
    _cosine,
)
from matgraph.graph_store import KnowledgeBase
from matgraph.retriever import Hop, node_label
from matgraph.schema import Material

logger = logging.getLogger(__name__)

USER_MATERIALS_PATH = DATA_DIR / "_user_materials.json"
# similar_to needs the five core computed properties (band gap, formation
# energy, energy above hull, density, magnetization): every Materials Project
# record has them, a composition-only entry has none. Elastic/dielectric
# dimensions (~20% coverage) are optional and mean-imputed, as in the build.
MIN_SIMILARITY_DIMENSIONS = 5
RELATION_FIELDS = [
    "contains", "has_space_group", "member_of", "classified_as", "suitable_for",
    "similar_to", "has_oxidation_state", "has_formula_pattern",
]

Provenance = Literal["mp", "derived", "guessed", "predicted", "unavailable"]
# Exact properties of a user-supplied structure (not computed by DFT).
STRUCTURE_DERIVED = {"density", "volume", "nsites"}

REPORTED_PROPERTIES = [
    "band_gap", "is_gap_direct", "is_metal", "formation_energy_per_atom", "energy_above_hull",
    "is_stable", "density", "volume", "nsites", "total_magnetization", "ordering", "theoretical",
    "bulk_modulus_vrh", "shear_modulus_vrh", "universal_anisotropy", "homogeneous_poisson",
    "dielectric_total", "dielectric_electronic", "dielectric_ionic", "refractive_index",
]


class IngestError(ValueError):
    pass


# --- property sources ---------------------------------------------------------


class PropertySource(Protocol):
    name: str

    def fetch(self, key: str) -> dict[str, Any] | None:
        """Return an MP-summary-shaped raw record, or None if unknown."""


class MPSource:
    name = "mp"

    def fetch(self, key: str) -> dict[str, Any] | None:
        from matgraph.mp_client import fetch_ground_state_by_formula, fetch_material_by_id

        if re.fullmatch(r"mp-\w+|mvc-\w+", key.strip()):
            return fetch_material_by_id(key.strip())
        return fetch_ground_state_by_formula(key.strip())


def _species_string(symbol: str, oxi: float) -> str | None:
    if not float(oxi).is_integer():
        return None
    v = int(oxi)
    magnitude = "" if abs(v) == 1 else str(abs(v))
    return f"{symbol}{magnitude}{'+' if v >= 0 else '-'}"


class CachedRecordSource:
    """Serves an already-fetched MP record (from data/raw/) - no API call.
    Used to replay persisted user materials after a rebuild, and by the
    leave-one-out ingestion evaluation."""

    name = "mp"

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw

    def fetch(self, key: str) -> dict[str, Any] | None:
        return dict(self.raw)


class CompositionSource:
    """Composition-only record for a formula with no Materials Project entry."""

    name = "composition"

    def fetch(self, key: str) -> dict[str, Any] | None:
        from pymatgen.core import Composition

        try:
            comp = Composition(key.strip())
        except Exception as error:  # noqa: BLE001 - pymatgen raises several types
            raise IngestError(f"{key!r} is not a valid chemical formula: {error}") from error
        if comp.num_atoms == 0:
            raise IngestError(f"{key!r} is not a valid chemical formula")
        elements = sorted(el.symbol for el in comp.elements)
        species: list[str] = []
        try:
            guesses = comp.oxi_state_guesses() if comp.num_atoms <= 60 else []
        except Exception:  # noqa: BLE001
            guesses = []
        if guesses:
            species = [s for el, v in guesses[0].items() if (s := _species_string(el, v))]
        reduced = comp.reduced_formula
        return {
            "material_id": f"user-{reduced}",
            "formula_pretty": reduced,
            "elements": elements,
            "chemsys": "-".join(elements),
            "formula_anonymous": comp.anonymized_formula,
            "possible_species": species,
            "_species_guessed": True,
            "_fractions": {el.symbol: round(frac, 4) for el, frac in comp.fractional_composition.items()},
            "description": f"{reduced}: composition-only entry (no Materials Project record).",
        }


# --- report models -----------------------------------------------------------------


class PropertyValue(BaseModel):
    name: str
    value: Any = None
    provenance: Provenance
    # Predicted values only: units, model and the expected error measured on
    # unseen graph materials (artifacts/eval/cgcnn_graph_eval.md).
    units: str | None = None
    model: str | None = None
    expected_error: float | None = None
    expected_error_units: str | None = None


class RuleEvaluation(BaseModel):
    domain: str
    rule: str
    status: Literal["fired", "not_fired", "cannot_evaluate"]
    reason: str = ""
    predicted: bool = False  # evaluated on CGCNN predictions, not DFT values


class Neighbour(BaseModel):
    material_id: str
    formula: str
    weight: float
    kind: Literal["similar_to", "composition"]


class SharedHub(BaseModel):
    node: str
    relation: str
    shared_with: int  # existing materials connected to this hub
    new_node: bool = False
    predicted: bool = False  # edge rests on a CGCNN prediction


class RelationReport(BaseModel):
    input: str
    source: Literal["mp", "composition", "cgcnn", "graph"]
    material_id: str
    formula: str
    node_id: str
    already_in_graph: bool = False
    persisted: bool = False
    properties: list[PropertyValue] = Field(default_factory=list)
    edges: list[Hop] = Field(default_factory=list)
    shared_hubs: list[SharedHub] = Field(default_factory=list)
    similar_materials: list[Neighbour] = Field(default_factory=list)
    compositional_neighbours: list[Neighbour] = Field(default_factory=list)
    rules: list[RuleEvaluation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# --- helpers ----------------------------------------------------------------------------


def _edge_targets(dp: Any) -> list[tuple[str, Any, dict]]:
    """(relation, target DataPoint, edge props) for every relationship field."""
    out = []
    for field in RELATION_FIELDS:
        value = getattr(dp, field, None)
        if value is None:
            continue
        items = value if isinstance(value, list) else [value]
        for item in items:
            props: dict[str, Any] = {}
            target = item
            if isinstance(item, tuple):
                edge, target = item
                if getattr(edge, "weight", None) is not None:
                    props["weight"] = edge.weight
                if getattr(edge, "properties", None):
                    props["properties"] = dict(edge.properties)
            if isinstance(target, list):
                out.extend((field, t, props) for t in target)
            else:
                out.append((field, target, props))
    return out


def _node_props(dp: Any) -> dict[str, Any]:
    """Scalar properties of a DataPoint, in the shape get_graph_data() returns."""
    props: dict[str, Any] = {"type": type(dp).__name__}
    for name, value in dp.__dict__.items():
        if name in {"metadata", "id"}:
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            props[name] = value
    return props


def _hop_props(edge_props: dict[str, Any]) -> dict[str, Any]:
    """Rule text, weight and (for CGCNN-based edges) provenance of an edge."""
    from matgraph.graph_store import edge_annotations

    inner = edge_annotations(edge_props)
    picked = {"rule": inner.get("rule"), "weight": edge_props.get("weight"),
              "provenance": inner.get("provenance"), "model": inner.get("model")}
    return {k: v for k, v in picked.items() if v is not None}


def _relation_stub(material: Material) -> Material:
    """Copy of an existing Material with no relationship fields, used as a
    similar_to target when persisting: upserts identical properties without
    re-adding (or dropping) any of its existing edges."""
    stub = material.model_copy()
    for field in RELATION_FIELDS:
        object.__setattr__(stub, field, None)
    return stub


def _fractions(formula: str) -> dict[str, float]:
    from pymatgen.core import Composition

    try:
        return {el.symbol: frac for el, frac in Composition(formula).fractional_composition.items()}
    except Exception:  # noqa: BLE001
        return {}


def _fraction_cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(a[k] * b.get(k, 0.0) for k in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def evaluate_rules(raw: dict[str, Any], config: dict[str, Any]) -> list[RuleEvaluation]:
    """Every application rule with fired / not_fired / cannot_evaluate.

    For a composition-only record, compositional conditions are still
    checked (a rule whose compositional part already fails is not_fired);
    anything needing a computed property is cannot_evaluate.
    """
    rules = {r["domain"]: r["rule"] for r in config["application_rules"]}
    if raw.get("band_gap") is not None and raw.get("energy_above_hull") is not None:
        fired = {d for d, _ in apply_application_rules(raw, config)}
        return [RuleEvaluation(domain=d, rule=r, status="fired" if d in fired else "not_fired") for d, r in rules.items()]
    if raw.get("predicted_band_gap") is not None:
        return _predicted_rule_evaluations(raw, config, rules)

    elements = set(raw["elements"])
    tms = set(config["transition_metals"])
    out = []
    for domain, rule in rules.items():
        if domain == "battery_cathode" and not ("Li" in elements and elements & tms):
            out.append(RuleEvaluation(domain=domain, rule=rule, status="not_fired",
                                      reason="composition lacks Li or a transition metal"))
        elif domain == "photocatalyst" and "O" not in elements:
            out.append(RuleEvaluation(domain=domain, rule=rule, status="not_fired", reason="contains no O"))
        else:
            needs = {"battery_cathode": "energy_above_hull", "magnetic_material": "ordering and total_magnetization"}
            out.append(RuleEvaluation(domain=domain, rule=rule, status="cannot_evaluate",
                                      reason=f"missing {needs.get(domain, 'band_gap')} (no Materials Project record)"))
    return out


def _interval_status(lo: float, hi: float, low: float, high: float) -> str:
    """Rule low <= x <= high for x anywhere in [lo, hi]: fired / not_fired / uncertain."""
    if low <= lo and hi <= high:
        return "fired"
    if hi < low or lo > high:
        return "not_fired"
    return "uncertain"


def _predicted_rule_evaluations(raw: dict[str, Any], config: dict[str, Any],
                                rules: dict[str, str]) -> list[RuleEvaluation]:
    """Rules on CGCNN predictions. Only the band gap and metallicity are
    predicted, and the gap carries its expected error: a rule fires only if
    it holds across gap ± error, is not_fired only if it fails across it, and
    cannot be evaluated when the interval straddles a threshold or the rule
    needs a property CGCNN does not predict (E_hull, magnetization)."""
    from matgraph.cgcnn.interpret import expected_errors

    gap, metal = raw["predicted_band_gap"], raw.get("predicted_is_metal")
    err = (expected_errors().get("predicted_band_gap") or {}).get("value") or 0.0
    lo, hi = gap - err, gap + err
    gap_text = f"predicted band gap {gap:.2f} ± {err:.2f} eV"
    elements = set(raw["elements"])
    out = []

    def add(domain, status, reason, predicted=False):
        # predicted marks verdicts that rest on the CGCNN values; compositional
        # verdicts and "needs a property CGCNN lacks" are exact.
        out.append(RuleEvaluation(domain=domain, rule=rules[domain], status=status, reason=reason, predicted=predicted))

    def by_gap(domain, low, high):
        status = _interval_status(lo, hi, low, high)
        if status == "uncertain":
            add(domain, "cannot_evaluate", f"{gap_text} straddles the rule threshold", predicted=True)
        else:
            add(domain, status, gap_text, predicted=True)

    if "Li" in elements and elements & set(config["transition_metals"]):
        add("battery_cathode", "cannot_evaluate", "needs energy_above_hull (CGCNN predicts formation energy only)")
    else:
        add("battery_cathode", "not_fired", "composition lacks Li or a transition metal")
    if metal:
        add("semiconductor_device", "not_fired",
            f"predicted metallic (P(metal) = {raw.get('predicted_metal_probability', 0):.2f})", predicted=True)
    else:
        by_gap("semiconductor_device", 1e-9, 3.5)
    if "O" not in elements:
        add("photocatalyst", "not_fired", "contains no O")
    else:
        by_gap("photocatalyst", 1.5, 3.5)
    if hi <= 2:
        add("dielectric", "not_fired", gap_text, predicted=True)
    else:
        add("dielectric", "cannot_evaluate", "needs total_magnetization (not predicted)")
    add("magnetic_material", "cannot_evaluate", "needs magnetic ordering and magnetization (not predicted)")
    return [r for d in rules for r in out if r.domain == d]


def predicted_properties(values: dict[str, Any]) -> list[PropertyValue]:
    """PropertyValues for the predicted_* attributes present in `values` (a raw
    record or a graph node's props), with model and expected error."""
    from matgraph.cgcnn.interpret import expected_errors
    from matgraph.schema import Material as _Material

    errors = expected_errors()
    units = {"predicted_formation_energy_per_atom": "eV/atom", "predicted_band_gap": "eV", "predicted_efermi": "eV",
             "predicted_bulk_modulus_vrh": "GPa", "predicted_shear_modulus_vrh": "GPa"}
    out = []
    for name in _Material.model_fields:
        if not name.startswith("predicted_") or values.get(name) is None:
            continue
        err = errors.get(name) or {}
        out.append(PropertyValue(
            name=name, value=values[name], provenance="predicted", units=units.get(name),
            model=values.get("prediction_model"), expected_error=err.get("value"),
            expected_error_units=err.get("units"),
        ))
    return out


def _properties(raw: dict[str, Any], source: str, material: Material) -> list[PropertyValue]:
    out = []
    for name in ["formula_pretty", "chemsys", "formula_anonymous"]:
        out.append(PropertyValue(name=name, value=raw.get(name), provenance="mp" if source == "mp" else "derived"))
    if source == "composition":
        out.append(PropertyValue(name="atomic_fractions", value=raw.get("_fractions"), provenance="derived"))
    out.append(PropertyValue(
        name="oxidation_states", value=raw.get("possible_species") or None,
        provenance=("guessed" if raw.get("_species_guessed") else "mp") if raw.get("possible_species") else "unavailable",
    ))
    if raw.get("symmetry"):
        sym = raw["symmetry"]
        out.append(PropertyValue(name="space_group", value=f"{sym['symbol']} (#{sym['number']}, {sym['crystal_system']})",
                                 provenance="mp" if source == "mp" else "derived"))
    else:
        out.append(PropertyValue(name="space_group", value=None, provenance="unavailable"))
    for name in REPORTED_PROPERTIES:
        value = getattr(material, name, None)
        provenance = "unavailable" if value is None else ("derived" if source == "cgcnn" else "mp")
        out.append(PropertyValue(name=name, value=value, provenance=provenance))
    if source == "cgcnn":
        out.extend(predicted_properties(material.__dict__))
    return out


def _record_user_material(material_id: str, formula: str, source: str, key: str) -> None:
    entries = json.loads(USER_MATERIALS_PATH.read_text()) if USER_MATERIALS_PATH.exists() else []
    if not any(e["material_id"] == material_id for e in entries):
        entries.append({"material_id": material_id, "formula": formula, "source": source, "input": key})
        USER_MATERIALS_PATH.write_text(json.dumps(entries, indent=2))


# --- main entry point ----------------------------------------------------------------


async def ingest_material(
    kb: KnowledgeBase,
    key: str,
    persist: bool = False,
    sources: list[PropertySource] | None = None,
    curated: list[dict[str, Any]] | None = None,
    stats: dict[str, tuple[float, float]] | None = None,
) -> RelationReport:
    """Resolve `key` (MP id or formula), build the node, relate it, report."""
    key = key.strip()
    if not key:
        raise IngestError("empty material id / formula")
    g = kb.graph
    config = load_config()
    elements_table = load_elements_table()

    existing = g.material(key)
    if existing is None and not key.startswith("mp-"):
        try:
            from pymatgen.core import Composition

            existing = g.material(Composition(key).reduced_formula)
        except Exception:  # noqa: BLE001
            existing = None
    if existing is not None:
        return relation_report_for_existing(kb, existing, key)
    if key.startswith("struct-"):
        raise IngestError(f"{key!r} is not in the graph; add a structure with its CIF/POSCAR file instead")

    sources = sources or [MPSource(), CompositionSource()]
    raw, source_name = None, None
    for source in sources:
        if source.name == "composition" and key.startswith("mp-"):
            continue
        raw = source.fetch(key)
        if raw:
            source_name = source.name
            break
    if raw is None:
        raise IngestError(f"{key!r} not found in Materials Project")

    notes: list[str] = []
    if source_name == "mp":
        # The MP ground state for a formula may be a formula already in the graph
        # (e.g. input "Fe2O3" written differently), or the id may be one we have.
        already = g.material(raw["material_id"]) or g.material(raw["formula_pretty"])
        if already is not None:
            return relation_report_for_existing(kb, already, key)
        cache_file = DATA_DIR / f"{raw['material_id']}.json"
        if not cache_file.exists():  # cached (with its structure) for Phase 3's CGCNN
            cache_file.write_text(json.dumps(raw, indent=2))
        notes.append(f"Fetched {raw['material_id']} from Materials Project "
                     "(lowest energy-above-hull polymorph)" + (" for this formula." if not key.startswith("mp-") else "."))
    else:
        notes.append(f"{raw['formula_pretty']} has no Materials Project entry: only composition-derived facts are "
                     "known; computed properties are unavailable (a crystal structure would go to CGCNN in Phase 3).")

    for sym in raw["elements"]:
        if sym not in elements_table:
            raise IngestError(f"unknown element {sym!r}")

    factory = NodeFactory(config, elements_table)
    material = factory.material(raw, origin="user")
    material.property_source = source_name
    if source_name == "composition":
        material.profile = (
            f"{raw['formula_pretty']} is a composition-only entry with no Materials Project record, "
            f"in the {raw['chemsys']} chemical system, formula pattern {raw['formula_anonymous']}. "
            f"Contains {', '.join(elements_table[s]['name'] for s in raw['elements'])}. "
            "Band gap, stability, formation energy, density and all other computed properties are not available."
        )
    return await relate_and_report(kb, key, raw, source_name, material, factory, config, notes, persist, curated, stats)


async def relate_and_report(
    kb: KnowledgeBase,
    key: str,
    raw: dict[str, Any],
    source_name: str,
    material: Material,
    factory: NodeFactory,
    config: dict[str, Any],
    notes: list[str],
    persist: bool,
    curated: list[dict[str, Any]] | None = None,
    stats: dict[str, tuple[float, float]] | None = None,
) -> RelationReport:
    """Relate a built Material to the graph, report it, and preview or persist it.
    Shared by every property source (ingest_material, ingest_structure)."""
    g = kb.graph
    # similar_to: same 10-d cosine, on the frozen build-time scale.
    similar: list[Neighbour] = []
    stubs: list[Material] = []
    dims = count_similarity_dimensions(raw)
    if dims >= MIN_SIMILARITY_DIMENSIONS:
        stats = stats or load_similarity_stats()
        curated = curated if curated is not None else load_raw_materials(config)
        vec = similarity_vector(raw, stats)
        scored = sorted(
            ((other, _cosine(vec, similarity_vector(other, stats))) for other in curated),
            key=lambda pair: -pair[1],
        )
        top_k, min_w = config["similarity"]["top_k"], config["similarity"]["min_edge_weight"]
        chosen = [(o, w) for o, w in scored if w >= min_w and g.material(o["material_id"])][:top_k]
        from matgraph.schema import Edge

        edges = []
        for other, weight in chosen:
            stub = _relation_stub(factory.material(other))
            stubs.append(stub)
            edges.append((Edge(relationship_type="similar_to", weight=weight), stub))
            similar.append(Neighbour(material_id=other["material_id"], formula=other["formula_pretty"],
                                     weight=round(weight, 4), kind="similar_to"))
        material.similar_to = edges
    elif source_name != "cgcnn":  # ingest_structure explains its own case
        notes.append(f"No similar_to edges: only {dims}/10 DFT similarity dimensions have values "
                     f"(needs {MIN_SIMILARITY_DIMENSIONS}); related by composition instead.")

    # Compositional neighbours: cosine over element-fraction vectors.
    fractions = raw.get("_fractions") or _fractions(raw["formula_pretty"])
    comp_scores = []
    for nid, props in g.nodes_of_type("Material"):
        if nid in g.overlay_nodes:
            continue
        comp_scores.append((props, _fraction_cosine(fractions, _fractions(props["formula"]))))
    comp_scores.sort(key=lambda pair: -pair[1])
    compositional = [
        Neighbour(material_id=p["material_id"], formula=p["formula"], weight=round(w, 4), kind="composition")
        for p, w in comp_scores[:5] if w > 0
    ]

    # Relate: shared hubs (existing nodes reused) vs new nodes.
    node_id = str(material.id)
    material_props = _node_props(material)
    edges_out: list[Hop] = []
    shared: list[SharedHub] = []
    overlay_edges: list[tuple[str, str, str, dict]] = []
    new_nodes: list[tuple[str, dict]] = []
    for rel, target, props in _edge_targets(material):
        tid = str(target.id)
        is_new = tid not in g.node
        tprops = _node_props(target)
        if is_new:
            new_nodes.append((tid, tprops))
            # A new SpaceGroup also brings its crystal_system edge.
            if rel == "has_space_group":
                cs = target.crystal_system
                if str(cs.id) not in g.node:
                    new_nodes.append((str(cs.id), _node_props(cs)))
                overlay_edges.append((tid, str(cs.id), "crystal_system", {}))
        overlay_edges.append((node_id, tid, rel, props))
        hop_props = _hop_props(props)
        edges_out.append(Hop(source=node_label({**material_props}), relation=rel,
                             target=node_label(tprops if is_new else g.node[tid]), properties=hop_props))
        if rel != "similar_to":
            degree = 0 if is_new else sum(
                1 for r, sid, _ in g.in_[tid] if r == rel and g.node[sid].get("type") == "Material"
            )
            shared.append(SharedHub(node=node_label(tprops if is_new else g.node[tid]), relation=rel,
                                    shared_with=degree, new_node=is_new,
                                    predicted=hop_props.get("provenance") == "predicted"))
    # Most selective shared hubs first; new nodes last.
    shared.sort(key=lambda h: (h.new_node, h.shared_with == 0, h.shared_with))

    report = RelationReport(
        input=key, source=source_name, material_id=raw["material_id"], formula=raw["formula_pretty"],
        node_id=node_id, properties=_properties(raw, source_name, material), edges=edges_out,
        shared_hubs=shared, similar_materials=similar, compositional_neighbours=compositional,
        rules=evaluate_rules(raw, config), notes=notes,
    )
    if new_nodes:
        report.notes.append("New graph nodes created: " + ", ".join(node_label(p) for _, p in new_nodes) + ".")

    if persist:
        from matgraph.graph_store import load_graph
        from matgraph.pipeline import add_to_graph

        await add_to_graph([material] + stubs)
        _record_user_material(raw["material_id"], raw["formula_pretty"], source_name, key)
        kb.graph = await load_graph()
        report.persisted = True
        report.node_id = str(material.id)
    else:
        for nid, props in new_nodes:
            if nid not in g.node:
                g.add_overlay_node(nid, props)
        g.add_overlay_node(node_id, material_props)
        for s, t, rel, props in overlay_edges:
            g.add_overlay_edge(s, t, rel, props)
        report.notes.append("Preview only: added to this session's in-memory graph, not saved (use persist).")
    return report


def relation_report_for_existing(kb: KnowledgeBase, node_id: str, key: str) -> RelationReport:
    """Relations of a material that is already in the graph (no ingestion needed)."""
    g = kb.graph
    props = g.node[node_id]
    edges, shared, similar = [], [], []
    for rel, tid, eprops in g.out[node_id]:
        edges.append(Hop(source=node_label(props), relation=rel, target=node_label(g.node[tid]),
                         properties=_hop_props(eprops)))
        if rel == "similar_to":
            similar.append(Neighbour(material_id=g.node[tid]["material_id"], formula=g.node[tid]["formula"],
                                     weight=round(eprops.get("weight") or 0, 4), kind="similar_to"))
        else:
            degree = sum(1 for r, sid, _ in g.in_[tid] if r == rel and g.node[sid].get("type") == "Material") - 1
            shared.append(SharedHub(node=node_label(g.node[tid]), relation=rel, shared_with=degree,
                                    predicted=_hop_props(eprops).get("provenance") == "predicted"))
    shared.sort(key=lambda h: (h.shared_with == 0, h.shared_with))
    raw_like = {k: props.get(k) for k in ("band_gap", "energy_above_hull", "is_metal", "total_magnetization", "ordering",
                                          "predicted_band_gap", "predicted_is_metal", "predicted_metal_probability")}
    raw_like["elements"] = [p["symbol"] for _, p in g.targets(node_id, "contains")]
    known = "derived" if props.get("property_source") == "cgcnn" else "mp"
    has_rule_inputs = raw_like["band_gap"] is not None or raw_like["predicted_band_gap"] is not None
    return RelationReport(
        input=key, source="graph", material_id=props["material_id"], formula=props["formula"], node_id=node_id,
        already_in_graph=True,
        properties=[PropertyValue(name=n, value=props.get(n), provenance=known if props.get(n) is not None else "unavailable")
                    for n in REPORTED_PROPERTIES] + predicted_properties(props),
        edges=edges, shared_hubs=shared, similar_materials=similar,
        rules=evaluate_rules(raw_like, load_config()) if has_rule_inputs else [],
        notes=[f"{props['formula']} ({props['material_id']}) is already in the graph; showing its existing relations."],
    )


# --- explanation ------------------------------------------------------------------------

_PROVENANCE_SOURCE = {
    "mp": "Materials Project summary",
    "derived": "derived from the formula (pymatgen Composition)",
    "guessed": "guessed from the formula or structure (pymatgen; not Materials Project data)",
    "predicted": "CGCNN prediction (pre-trained txie-93/cgcnn model; not a DFT value)",
}


def relation_evidence(report: RelationReport):
    """The relation report as an evidence bundle (same format as query answers)."""
    from matgraph.evidence import EvidenceBundle, EvidenceFact

    facts: list[EvidenceFact] = []

    def add(kind, text, source, material_id=None, formula=None, predicted=False, **values):
        facts.append(EvidenceFact(id=f"E{len(facts) + 1}", kind=kind, material_id=material_id, formula=formula,
                                  text=text, source=source, values=values, predicted=predicted))

    f, mid = report.formula, report.material_id
    for p in report.properties:
        if p.provenance == "unavailable" or p.value is None:
            continue
        value = p.value if not isinstance(p.value, float) else round(p.value, 4)
        numeric = {p.name: p.value} if isinstance(p.value, (int, float)) and not isinstance(p.value, bool) else {}
        if p.provenance == "predicted":
            error = "" if p.expected_error is None else (
                f"; expected error ±{p.expected_error:.3g} {p.expected_error_units or ''}".rstrip())
            if p.expected_error is not None:
                numeric["expected_error"] = p.expected_error
            add("property", f"{f} {p.name} ≈ {value}{(' ' + p.units) if p.units else ''} (predicted by {p.model}{error}).",
                _PROVENANCE_SOURCE["predicted"], mid, f, predicted=True, **numeric)
            continue
        add("property", f"{f} {p.name} = {value}.", _PROVENANCE_SOURCE[p.provenance], mid, f, **numeric)
    missing = [p.name for p in report.properties if p.provenance == "unavailable"]
    if missing:
        add("missing", f"{f}: NOT AVAILABLE: {', '.join(missing)}.", "no Materials Project (DFT) value", mid, f,
            missing=missing)
    for hub in report.shared_hubs:
        status = "a new node in the graph" if hub.new_node else f"shared with {hub.shared_with} existing materials"
        basis = ", based on a CGCNN prediction" if hub.predicted else ""
        add("edge", f"{f} -{hub.relation}-> {hub.node} ({status}{basis}).",
            f"graph edge {hub.relation}" + (" (predicted)" if hub.predicted else ""), mid, f,
            predicted=hub.predicted, shared_with=hub.shared_with)
    for n in report.similar_materials:
        add("similarity", f"{f} -similar_to-> {n.formula} ({n.material_id}) weight {n.weight}.",
            "similar_to: 10-d property cosine (frozen build-time scale)", n.material_id, n.formula, weight=n.weight)
    for n in report.compositional_neighbours:
        add("similarity", f"{f} has element-fraction cosine {n.weight} with {n.formula} ({n.material_id}).",
            "composition similarity", n.material_id, n.formula, weight=n.weight)
    for r in report.rules:
        add("edge", f"{f} rule {r.domain}: {r.status}" + (f" ({r.reason})" if r.reason else "") + f' - "{r.rule}".',
            "application-domain rule evaluation" + (" on CGCNN predictions" if r.predicted else ""), mid, f,
            predicted=r.predicted)
    # The material itself first, so an explanation covers its own facts (and,
    # for a structure, its predictions), then the materials it relates to.
    related = list(dict.fromkeys([mid] + [n.material_id for n in report.similar_materials
                                          + report.compositional_neighbours]))
    return EvidenceBundle(facts=facts, material_ids=related)


def explain_relation(kb: KnowledgeBase, report: RelationReport, use_llm: bool = True):
    """(Explanation, GroundingReport, generator, model) for a relation report."""
    from matgraph.generator import explain_with_llm, explain_with_template
    from matgraph.grounding import check_explanation
    from matgraph.llm import LLMUnavailable, llm_configured
    from matgraph.query_schema import QueryConstraints

    evidence = relation_evidence(report)
    question = f"How does {report.formula} relate to the materials already in the knowledge graph?"
    generator, model = "template", None
    if use_llm and llm_configured():
        try:
            explanation, model = explain_with_llm(question, QueryConstraints(), evidence)
            generator = "llm"
        except LLMUnavailable as error:
            logger.warning("LLM unavailable for relation explanation: %s", error)
            explanation = explain_with_template(evidence)
    else:
        explanation = explain_with_template(evidence)
    grounding = check_explanation(explanation, evidence, kb.graph, evidence.material_ids + [report.material_id],
                                  question)
    return explanation, grounding, generator, model


async def replay_user_materials(kb: KnowledgeBase) -> list[str]:
    """Re-add persisted user materials after a full rebuild (which prunes
    everything), from the local cache - so `persist` survives rebuilds."""
    if not USER_MATERIALS_PATH.exists():
        return []
    replayed = []
    for entry in json.loads(USER_MATERIALS_PATH.read_text()):
        if entry["source"] == "cgcnn":
            from matgraph.ingest_structure import USER_STRUCTURES_DIR, ingest_structure

            record_file = USER_STRUCTURES_DIR / f"{entry['material_id']}.json"
            if not record_file.exists():
                logger.warning("cannot replay %s: %s missing", entry["material_id"], record_file)
                continue
            # From the stored predictions, so a rebuild reproduces the node exactly.
            await ingest_structure(kb, label=entry.get("input") or entry["material_id"], persist=True,
                                   record=json.loads(record_file.read_text()))
            replayed.append(entry["formula"])
            continue
        if entry["source"] == "mp":
            cache_file = DATA_DIR / f"{entry['material_id']}.json"
            if not cache_file.exists():
                logger.warning("cannot replay %s: %s missing", entry["material_id"], cache_file)
                continue
            sources: list[PropertySource] = [CachedRecordSource(json.loads(cache_file.read_text()))]
            key = entry["material_id"]
        else:
            sources, key = [CompositionSource()], entry["formula"]
        await ingest_material(kb, key, persist=True, sources=sources)
        replayed.append(entry["formula"])
    return replayed
