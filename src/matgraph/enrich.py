"""Deterministic enrichment: raw MP records -> Cognee DataPoint graph.

No LLM calls anywhere in this module. Property classification, application
domain assignment, and material-material similarity are all rule-based so
every edge in the resulting graph is traceable to a concrete threshold or
formula (see config/materials.yaml).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import yaml

from matgraph.schema import (
    ApplicationDomain,
    ChemicalSystem,
    CrystalSystem,
    Edge,
    Element,
    FormulaPattern,
    Material,
    OxidationState,
    PropertyClass,
    SpaceGroup,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config" / "materials.yaml"
ELEMENTS_PATH = ROOT / "config" / "elements.json"
DATA_DIR = ROOT / "data" / "raw"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text())


def load_elements_table() -> dict[str, dict[str, Any]]:
    return json.loads(ELEMENTS_PATH.read_text())


# --- record validation ----------------------------------------------------

# Every field a Material node needs to exist at all: schema.Material declares
# each of these non-optional (no default), and apply_application_rules reads
# several of them directly off the raw record.
#
# A record missing any of them is skipped, not defaulted. Substituting 0.0 for
# an absent band_gap would classify that material as a metal, and substituting
# 0.0 for an absent energy_above_hull would classify it as perfectly stable -
# both would silently corrupt every downstream query rather than failing
# visibly. Skipping loses one material; defaulting poisons the graph.
REQUIRED_RAW_FIELDS = (
    "material_id",
    "formula_pretty",
    "chemsys",
    "elements",
    "symmetry",
    "band_gap",
    "is_gap_direct",
    "is_metal",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "density",
    "volume",
    "nsites",
    "total_magnetization",
    "ordering",
    "theoretical",
)

_SYMMETRY_SUBFIELDS = ("symbol", "number", "crystal_system")


def _missing_fields(raw: dict[str, Any]) -> list[str]:
    """Names of required fields absent from a raw MP record.

    Uses `is None` rather than falsiness so legitimate values - False for
    is_metal, 0.0 for the band gap of a metal - are not mistaken for missing.
    """
    missing = [field for field in REQUIRED_RAW_FIELDS if raw.get(field) is None]

    # elements must be non-empty: every Material is expected to have at least
    # one `contains` edge, and verify.py asserts exactly that.
    if raw.get("elements") is not None and not raw["elements"]:
        missing.append("elements (empty)")

    symmetry = raw.get("symmetry")
    if symmetry is not None:
        missing += [
            f"symmetry.{sub}" for sub in _SYMMETRY_SUBFIELDS if symmetry.get(sub) is None
        ]

    return missing


def filter_valid_materials(raw_materials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop raw records that can't produce a valid Material, loudly.

    Filtering happens before similarity is computed so that similar_to edges
    can never point at a material that was subsequently skipped.
    """
    valid: list[dict[str, Any]] = []
    for raw in raw_materials:
        missing = _missing_fields(raw)
        if missing:
            logger.warning(
                "skipping %s: missing required field(s): %s",
                raw.get("material_id", "<unknown material_id>"),
                ", ".join(missing),
            )
        else:
            valid.append(raw)

    skipped = len(raw_materials) - len(valid)
    if skipped:
        logger.warning(
            "skipped %d of %d cached records as incomplete; building graph from %d",
            skipped,
            len(raw_materials),
            len(valid),
        )
    return valid


def load_raw_materials(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load every cached material from data/raw/ (one JSON file per material_id,
    listed in _manifest.json - see matgraph.mp_client.fetch_and_cache_bulk)."""
    manifest_path = DATA_DIR / "_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no manifest at {manifest_path} (run scripts/fetch_materials.py first)"
        )
    material_ids = json.loads(manifest_path.read_text())
    return [json.loads((DATA_DIR / f"{mid}.json").read_text()) for mid in material_ids]


# --- classification -----------------------------------------------------


def _bucket(value: float | None, buckets: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Find the bucket `value` falls into, or None if it matches none of them.

    Returns None rather than raising. The buckets in config/materials.yaml are
    exhaustive over the values Materials Project actually returns, so a miss
    means one malformed record - a missing value, or a NaN (which fails every
    comparison and would otherwise fall through to a raise). Aborting the whole
    graph build over a single bad material is the wrong trade: the caller logs
    it and skips just that one classification. This also matches the existing
    convention in this module, where classify_ordering already returns None for
    input it can't classify.
    """
    if value is None or value != value:  # noqa: PLR0124 (NaN never equals itself)
        return None
    for bucket in buckets:
        lo = bucket.get("min", -math.inf)
        hi = bucket.get("max", math.inf)
        if lo <= value <= hi:
            return bucket
    return None


def classify_band_gap(band_gap: float | None, config: dict[str, Any]) -> dict[str, Any] | None:
    bucket = _bucket(band_gap, config["band_gap_buckets"])
    if bucket is None:
        logger.warning("band_gap %r matched no bucket; skipping band_gap classification", band_gap)
        return None
    return {
        "kind": "band_gap",
        "name": bucket["name"],
        "description": f"band_gap classified as '{bucket['name']}' (value={band_gap:.3f} eV)",
    }


def classify_stability(e_hull: float | None, config: dict[str, Any]) -> dict[str, Any] | None:
    bucket = _bucket(e_hull, config["stability_buckets"])
    if bucket is None:
        logger.warning(
            "energy_above_hull %r matched no bucket; skipping stability classification", e_hull
        )
        return None
    return {
        "kind": "stability",
        "name": bucket["name"],
        "description": (
            f"stability classified as '{bucket['name']}' "
            f"(energy_above_hull={e_hull:.4f} eV/atom)"
        ),
    }


def classify_ordering(ordering: str) -> dict[str, Any] | None:
    if not ordering or ordering == "Unknown":
        return None
    return {
        "kind": "magnetic_ordering",
        "name": ordering,
        "description": f"magnetic ordering reported by Materials Project as '{ordering}'",
    }


# --- application domain rules --------------------------------------------


def apply_application_rules(
    material: dict[str, Any], config: dict[str, Any]
) -> list[tuple[str, str]]:
    """Return [(domain_name, rule_text), ...] for every rule this material satisfies."""
    elements = set(material["elements"])
    transition_metals = set(config["transition_metals"])
    band_gap = material["band_gap"]
    is_metal = material["is_metal"]
    e_hull = material["energy_above_hull"]
    total_mag = abs(material["total_magnetization"])
    ordering = material["ordering"]

    matches: list[tuple[str, str]] = []
    rules_by_domain = {r["domain"]: r["rule"] for r in config["application_rules"]}

    if "Li" in elements and elements & transition_metals and e_hull <= 0.05:
        matches.append(("battery_cathode", rules_by_domain["battery_cathode"]))

    if 0 < band_gap <= 3.5 and not is_metal:
        matches.append(("semiconductor_device", rules_by_domain["semiconductor_device"]))

    if "O" in elements and 1.5 <= band_gap <= 3.5:
        matches.append(("photocatalyst", rules_by_domain["photocatalyst"]))

    if band_gap > 2 and total_mag < 0.1:
        matches.append(("dielectric", rules_by_domain["dielectric"]))

    if (ordering in {"FM", "FiM"} and total_mag >= 0.1) or ordering == "AFM":
        matches.append(("magnetic_material", rules_by_domain["magnetic_material"]))

    return matches


# --- similarity -----------------------------------------------------------

# Every field here is (a) intensive/size-normalized - comparable regardless
# of which unit cell or supercell a calculation happened to use, (b) on a
# physically universal scale - not referenced to a per-calculation internal
# zero, and (c) not redundant with another field already in the vector.
# Together they span electronic, thermodynamic, structural-density,
# magnetic, mechanical, and dielectric behaviour: a genuine materials
# fingerprint rather than an arbitrary handful of numbers.
#
# Deliberately excluded, with reasons:
#   volume, nsites, num_magnetic_sites - NOT intensive: they scale with the
#     specific unit cell/supercell a calculation happened to use, so two
#     calculations of the "same" material can disagree on these even when
#     every intensive property matches.
#   cbm, vbm, efermi - absolute energies referenced to each calculation's
#     own internal zero, not a universal scale comparable across different
#     materials. band_gap (= cbm - vbm) is the physically meaningful,
#     calculation-independent quantity, and is already included.
#   is_magnetic - a boolean function of total_magnetization, which is
#     already in the vector as a continuous value; including both would
#     double-count the same signal.
#   e_electronic, e_ionic, refractive_index (n) - derived from / strongly
#     correlated with dielectric_total (e_total = e_electronic + e_ionic;
#     n ~ sqrt(e_electronic)). Including all of them would overweight
#     "polarizability" relative to the other, independent properties.
SIMILARITY_FIELDS = [
    "band_gap",
    "formation_energy_per_atom",
    "energy_above_hull",
    "density",
    "total_magnetization",
    "bulk_modulus_vrh",
    "shear_modulus_vrh",
    "universal_anisotropy",
    "homogeneous_poisson",
    "dielectric_total",
]


def _similarity_value(raw: dict[str, Any], field: str) -> float | None:
    """Pull one SIMILARITY_FIELDS value out of a *raw* MP record. Most fields
    match the raw JSON key directly, but bulk/shear modulus arrive nested as
    {"voigt": .., "reuss": .., "vrh": ..} and dielectric_total's raw key is
    e_total (dielectric_total is the flattened name used on the Material
    DataPoint, built later in build_graph - not present on the raw dict)."""
    if field == "bulk_modulus_vrh":
        return _vrh(raw.get("bulk_modulus"))
    if field == "shear_modulus_vrh":
        return _vrh(raw.get("shear_modulus"))
    if field == "dielectric_total":
        return raw.get("e_total")
    return raw.get(field)


def _zscore_vectors(materials: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Z-score every similarity field, imputing missing values with the column
    mean so a missing value contributes neutrally (z=0) rather than skewing the
    comparison in either direction.

    Imputation matters here because coverage is low: elastic data is present
    for 18.0% of materials and dielectric for 19.9% (measured across the full
    794-material set). Dropping those materials, or treating missing as zero,
    would either shrink the graph drastically or fabricate signal - so a
    material with no elastic data still gets meaningful similar_to edges from
    its remaining dimensions.
    """
    columns = {f: [_similarity_value(m, f) for m in materials] for f in SIMILARITY_FIELDS}
    stats = {}
    for f, values in columns.items():
        present = [v for v in values if v is not None]
        mean = sum(present) / len(present) if present else 0.0
        variance = sum((v - mean) ** 2 for v in present) / len(present) if present else 0.0
        std = math.sqrt(variance) or 1.0
        stats[f] = (mean, std)

    vectors = {}
    for m in materials:
        vec = []
        for f in SIMILARITY_FIELDS:
            value = _similarity_value(m, f)
            mean, std = stats[f]
            vec.append(0.0 if value is None else (value - mean) / std)
        vectors[m["material_id"]] = vec
    return vectors


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def compute_similarity_edges(
    materials: list[dict[str, Any]], min_weight: float, top_k: int
) -> dict[str, list[tuple[str, float]]]:
    """Return {material_id: [(other_material_id, weight), ...]}: each material's
    top_k most similar materials above min_weight.

    A flat global threshold doesn't scale with dataset size - at a few hundred
    materials a 0.7 cosine cutoff over just 4 dimensions produces tens of
    thousands of pairs (most materials look "similar enough" to many others),
    turning similar_to into noise. Capping to each material's nearest
    neighbours keeps the edge density meaningful regardless of how many
    materials are in the graph.
    """
    vectors = _zscore_vectors(materials)
    ids = list(vectors.keys())
    edges: dict[str, list[tuple[str, float]]] = {i: [] for i in ids}
    for i, id_i in enumerate(ids):
        sims = []
        for j, id_j in enumerate(ids):
            if i == j:
                continue
            sim = _cosine(vectors[id_i], vectors[id_j])
            if sim >= min_weight:
                sims.append((id_j, sim))
        sims.sort(key=lambda pair: -pair[1])
        edges[id_i] = sims[:top_k]
    return edges


def _vrh(value: dict[str, float] | None) -> float | None:
    """Extract the Voigt-Reuss-Hill average from an elastic modulus dict
    (e.g. {"voigt": .., "reuss": .., "vrh": ..}); None when not computed."""
    return value.get("vrh") if value else None


# --- graph construction -----------------------------------------------------


def build_graph(
    raw_materials: list[dict[str, Any]],
    config: dict[str, Any],
    elements_table: dict[str, dict[str, Any]],
) -> list[Material]:
    """Build the full set of interconnected DataPoint nodes for all materials."""

    # Must happen before compute_similarity_edges below: similarity is keyed by
    # material_id, and an edge pointing at a skipped material would KeyError
    # when the similar_to edges are wired up at the end of this function.
    raw_materials = filter_valid_materials(raw_materials)

    element_cache: dict[str, Element] = {}
    crystal_system_cache: dict[str, CrystalSystem] = {}
    space_group_cache: dict[str, SpaceGroup] = {}
    chemsys_cache: dict[str, ChemicalSystem] = {}
    property_class_cache: dict[tuple[str, str], PropertyClass] = {}
    domain_cache: dict[str, ApplicationDomain] = {}
    oxidation_state_cache: dict[str, OxidationState] = {}
    formula_pattern_cache: dict[str, FormulaPattern] = {}

    def get_element(symbol: str) -> Element:
        if symbol not in element_cache:
            info = elements_table.get(symbol)
            if info is None:
                # Shouldn't happen: config/elements.json covers Z=1-103, which
                # spans every element Materials Project reports. A bare KeyError
                # here would be opaque, so name the fix instead.
                raise KeyError(
                    f"element {symbol!r} is not in config/elements.json - "
                    "regenerate it with scripts/generate_elements.py"
                )
            element_cache[symbol] = Element(
                symbol=symbol,
                name=info["name"],
                atomic_number=info["atomic_number"],
                group=info["group"],
                period=info["period"],
                electronegativity=info["electronegativity"],
                covalent_radius_pm=info["covalent_radius_pm"],
                category=info["category"],
            )
        return element_cache[symbol]

    def get_crystal_system(name: str) -> CrystalSystem:
        if name not in crystal_system_cache:
            crystal_system_cache[name] = CrystalSystem(name=name)
        return crystal_system_cache[name]

    def get_space_group(symbol: str, number: int, crystal_system_name: str) -> SpaceGroup:
        key = f"{symbol}:{number}"
        if key not in space_group_cache:
            space_group_cache[key] = SpaceGroup(
                symbol=symbol,
                number=number,
                crystal_system=get_crystal_system(crystal_system_name),
            )
        return space_group_cache[key]

    def get_chemsys(chemsys: str) -> ChemicalSystem:
        if chemsys not in chemsys_cache:
            chemsys_cache[chemsys] = ChemicalSystem(chemsys=chemsys)
        return chemsys_cache[chemsys]

    def get_property_class(kind: str, name: str, description: str) -> PropertyClass:
        key = (kind, name)
        if key not in property_class_cache:
            property_class_cache[key] = PropertyClass(kind=kind, name=name, description=description)
        return property_class_cache[key]

    def get_domain(name: str) -> ApplicationDomain:
        if name not in domain_cache:
            rules_by_domain = {r["domain"]: r["rule"] for r in config["application_rules"]}
            domain_cache[name] = ApplicationDomain(
                name=name, description=rules_by_domain.get(name, name)
            )
        return domain_cache[name]

    def get_oxidation_state(species: str) -> OxidationState:
        if species not in oxidation_state_cache:
            oxidation_state_cache[species] = OxidationState(species=species)
        return oxidation_state_cache[species]

    def get_formula_pattern(pattern: str) -> FormulaPattern:
        if pattern not in formula_pattern_cache:
            formula_pattern_cache[pattern] = FormulaPattern(pattern=pattern)
        return formula_pattern_cache[pattern]

    similarity_edges = compute_similarity_edges(
        raw_materials, config["similarity"]["min_edge_weight"], config["similarity"]["top_k"]
    )

    material_nodes: dict[str, Material] = {}
    for raw in raw_materials:
        contains = [get_element(sym) for sym in raw["elements"]]
        symmetry = raw["symmetry"]
        space_group = get_space_group(
            symmetry["symbol"], symmetry["number"], symmetry["crystal_system"]
        )
        chemsys = get_chemsys(raw["chemsys"])

        # Any of these can be None for a material whose value is missing or
        # unclassifiable; that material simply gets fewer classified_as edges
        # rather than failing the build (see _bucket).
        classifications = [
            c
            for c in (
                classify_band_gap(raw["band_gap"], config),
                classify_stability(raw["energy_above_hull"], config),
                classify_ordering(raw["ordering"]),
            )
            if c is not None
        ]
        classified_as = [
            get_property_class(c["kind"], c["name"], c["description"]) for c in classifications
        ]

        domain_matches = apply_application_rules(raw, config)
        suitable_for = [
            (Edge(relationship_type="suitable_for", properties={"rule": rule}), get_domain(name))
            for name, rule in domain_matches
        ]

        species = raw.get("possible_species") or []
        has_oxidation_state = [get_oxidation_state(sp) for sp in species]

        formula_pattern = raw.get("formula_anonymous")
        has_formula_pattern = (
            (Edge(relationship_type="has_formula_pattern"), get_formula_pattern(formula_pattern))
            if formula_pattern
            else None
        )

        material_nodes[raw["material_id"]] = Material(
            material_id=raw["material_id"],
            formula=raw["formula_pretty"],
            # robocrys text is optional - not every material has one, and the
            # key is absent entirely if the batch fetch skipped it.
            description=raw.get("description")
            or f"{raw['formula_pretty']} (no description available)",
            band_gap=raw["band_gap"],
            is_gap_direct=raw["is_gap_direct"],
            is_metal=raw["is_metal"],
            formation_energy_per_atom=raw["formation_energy_per_atom"],
            energy_above_hull=raw["energy_above_hull"],
            is_stable=raw["is_stable"],
            density=raw["density"],
            volume=raw["volume"],
            nsites=raw["nsites"],
            total_magnetization=raw["total_magnetization"],
            ordering=raw["ordering"],
            theoretical=raw["theoretical"],
            cbm=raw.get("cbm"),
            vbm=raw.get("vbm"),
            efermi=raw.get("efermi"),
            is_magnetic=raw.get("is_magnetic") or False,
            num_magnetic_sites=raw.get("num_magnetic_sites") or 0,
            bulk_modulus_vrh=_vrh(raw.get("bulk_modulus")),
            shear_modulus_vrh=_vrh(raw.get("shear_modulus")),
            universal_anisotropy=raw.get("universal_anisotropy"),
            homogeneous_poisson=raw.get("homogeneous_poisson"),
            dielectric_total=raw.get("e_total"),
            dielectric_electronic=raw.get("e_electronic"),
            dielectric_ionic=raw.get("e_ionic"),
            refractive_index=raw.get("n"),
            contains=contains,
            has_space_group=(Edge(relationship_type="has_space_group"), space_group),
            member_of=(Edge(relationship_type="member_of"), chemsys),
            classified_as=classified_as,
            suitable_for=suitable_for,
            has_oxidation_state=has_oxidation_state,
            has_formula_pattern=has_formula_pattern,
            # similar_to filled in below, once every Material node exists.
        )

    for material_id, material in material_nodes.items():
        pairs = similarity_edges.get(material_id, [])
        material.similar_to = [
            (
                Edge(relationship_type="similar_to", weight=weight),
                material_nodes[other_id],
            )
            for other_id, weight in pairs
        ]

    return list(material_nodes.values())
