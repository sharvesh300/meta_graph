"""Deterministic enrichment: raw MP records -> Cognee DataPoint graph.

No LLM calls anywhere in this module. Property classification, application
domain assignment, and material-material similarity are all rule-based so
every edge in the resulting graph is traceable to a concrete threshold or
formula (see config/materials.yaml).
"""

from __future__ import annotations

import json
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

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config" / "materials.yaml"
ELEMENTS_PATH = ROOT / "config" / "elements.json"
DATA_DIR = ROOT / "data" / "raw"
SIMILARITY_STATS_PATH = ROOT / "artifacts" / "similarity_stats.json"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text())


def load_elements_table() -> dict[str, dict[str, Any]]:
    return json.loads(ELEMENTS_PATH.read_text())


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


def _bucket(value: float, buckets: list[dict[str, Any]]) -> dict[str, Any]:
    for bucket in buckets:
        lo = bucket.get("min", -math.inf)
        hi = bucket.get("max", math.inf)
        if lo <= value <= hi:
            return bucket
    raise ValueError(f"value {value} did not match any bucket in {buckets}")


def classify_band_gap(band_gap: float, config: dict[str, Any]) -> dict[str, Any]:
    bucket = _bucket(band_gap, config["band_gap_buckets"])
    return {
        "kind": "band_gap",
        "name": bucket["name"],
        "description": f"band_gap classified as '{bucket['name']}' (value={band_gap:.3f} eV)",
    }


def classify_stability(e_hull: float, config: dict[str, Any]) -> dict[str, Any]:
    bucket = _bucket(e_hull, config["stability_buckets"])
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
    if field in ELASTIC_FIELDS:
        elastic = elastic_properties(raw)
        return elastic[field] if elastic else None
    if field == "dielectric_total":
        return raw.get("e_total")
    return raw.get(field)


SimilarityStats = dict[str, tuple[float, float]]  # field -> (mean, std)


def compute_similarity_stats(materials: list[dict[str, Any]]) -> SimilarityStats:
    """Per-field (mean, std) over the materials that have a value for it.

    Frozen at build time (see save_similarity_stats) so a material added later
    through matgraph.ingest is z-scored on the same scale as the existing
    graph instead of shifting it.
    """
    stats: SimilarityStats = {}
    for f in SIMILARITY_FIELDS:
        present = [v for v in (_similarity_value(m, f) for m in materials) if v is not None]
        mean = sum(present) / len(present) if present else 0.0
        variance = sum((v - mean) ** 2 for v in present) / len(present) if present else 0.0
        stats[f] = (mean, math.sqrt(variance) or 1.0)
    return stats


# Z-scores are clipped to +/- this many standard deviations. Several fields are
# heavy-tailed (universal_anisotropy runs 0 - 461: layered h-BN), and an
# unclipped z of ~180 on one dimension makes the cosine a function of that one
# field alone. Found in Phase 2's leave-one-out ingestion check.
Z_CLIP = 3.0


def similarity_vector(raw: dict[str, Any], stats: SimilarityStats) -> list[float]:
    """Z-scored similarity vector, clipped to +/- Z_CLIP. Missing values
    (elastic and dielectric data aren't computed for every material - roughly
    18% and 20% coverage) are imputed with the column mean, i.e. z=0, so a
    missing value contributes neutrally rather than skewing the comparison."""
    vec = []
    for f in SIMILARITY_FIELDS:
        value = _similarity_value(raw, f)
        mean, std = stats[f]
        z = 0.0 if value is None else (value - mean) / std
        vec.append(max(-Z_CLIP, min(Z_CLIP, z)))
    return vec


def count_similarity_dimensions(raw: dict[str, Any]) -> int:
    """How many SIMILARITY_FIELDS carry a real (non-imputed) value."""
    return sum(1 for f in SIMILARITY_FIELDS if _similarity_value(raw, f) is not None)


def save_similarity_stats(stats: SimilarityStats, path: Path = SIMILARITY_STATS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({f: list(v) for f, v in stats.items()}, indent=2))


def load_similarity_stats(path: Path = SIMILARITY_STATS_PATH) -> SimilarityStats:
    if not path.exists():
        raise FileNotFoundError(f"no similarity stats at {path} (run scripts/build_graph.py first)")
    return {f: (v[0], v[1]) for f, v in json.loads(path.read_text()).items()}


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
    stats = compute_similarity_stats(materials)
    vectors = {m["material_id"]: similarity_vector(m, stats) for m in materials}
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


ELASTIC_FIELDS = ("bulk_modulus_vrh", "shear_modulus_vrh", "universal_anisotropy", "homogeneous_poisson")


def elastic_status(raw: dict[str, Any]) -> str:
    """"ok", "missing", or "unphysical".

    Materials Project occasionally ships elastic tensors from mechanically
    unstable fits: e.g. Li(CuO2)2 with a bulk modulus of -300,604 GPa. A
    stable crystal must have positive bulk and shear moduli, and the
    universal anisotropy index is >= 0 by definition. Such tensors are
    discarded as a block - one bad component means the whole fit is
    unreliable - and reported as unphysical rather than silently used (they
    previously dominated the elastic z-scores in similar_to).
    """
    bulk, shear = _vrh(raw.get("bulk_modulus")), _vrh(raw.get("shear_modulus"))
    anisotropy = raw.get("universal_anisotropy")
    if bulk is None and shear is None:
        return "missing"
    if bulk is None or shear is None or bulk <= 0 or shear <= 0 or (anisotropy is not None and anisotropy < 0):
        return "unphysical"
    return "ok"


def elastic_properties(raw: dict[str, Any]) -> dict[str, float | None] | None:
    """The four elastic fields, or None when missing or unphysical."""
    if elastic_status(raw) != "ok":
        return None
    return {
        "bulk_modulus_vrh": _vrh(raw.get("bulk_modulus")),
        "shear_modulus_vrh": _vrh(raw.get("shear_modulus")),
        "universal_anisotropy": raw.get("universal_anisotropy"),
        "homogeneous_poisson": raw.get("homogeneous_poisson"),
    }


# --- profile text -----------------------------------------------------------

# (raw-record extractor, label, unit) for the optional properties a profile
# reports as either a value or "not available" - never silently omitted, so
# both semantic search and the LLM see what is missing.
_OPTIONAL_PROPERTIES = [
    (lambda r: (elastic_properties(r) or {}).get("bulk_modulus_vrh"), "bulk modulus", "GPa"),
    (lambda r: (elastic_properties(r) or {}).get("shear_modulus_vrh"), "shear modulus", "GPa"),
    (lambda r: r.get("e_total"), "total dielectric constant", ""),
    (lambda r: r.get("n"), "refractive index", ""),
]


def build_profile(
    raw: dict[str, Any],
    classifications: list[dict[str, Any]],
    domain_matches: list[tuple[str, str]],
    elements_table: dict[str, dict[str, Any]],
) -> str:
    """Deterministic one-paragraph property summary of a material.

    Built from the same raw record and rule outputs as the graph edges (no
    LLM), so every statement in it is also present as an attribute or edge.
    """
    classes = {c["kind"]: c["name"] for c in classifications}
    symmetry = raw["symmetry"]
    element_names = ", ".join(
        elements_table[sym]["name"] if sym in elements_table else sym for sym in raw["elements"]
    )
    gap_kind = "direct" if raw.get("is_gap_direct") else "indirect"
    gap_text = (
        "It is metallic (DFT band gap 0 eV)."
        if raw["band_gap"] == 0
        else f"It has a DFT band gap of {raw['band_gap']:.2f} eV "
        f"({gap_kind}, {classes.get('band_gap')})."
    )
    ordering = raw.get("ordering")
    ordering_text = (
        f"magnetic ordering {ordering}"
        if ordering and ordering != "Unknown"
        else "magnetic ordering not determined"
    )

    parts = [
        f"{raw['formula_pretty']} ({raw['material_id']}) is a {classes.get('stability')} material "
        f"(energy above hull {raw['energy_above_hull']:.3f} eV/atom) in the {raw['chemsys']} "
        f"chemical system, space group {symmetry['symbol']} "
        f"({symmetry['crystal_system'].lower()} crystal system).",
        gap_text,
        f"Formation energy {raw['formation_energy_per_atom']:.3f} eV/atom; "
        f"density {raw['density']:.2f} g/cm3.",
        f"Total magnetization {raw['total_magnetization']:.2f} bohr magnetons, {ordering_text}.",
        f"Contains {element_names}.",
    ]
    species = raw.get("possible_species") or []
    if species:
        parts.append(f"Oxidation states: {', '.join(species)}.")
    if raw.get("formula_anonymous"):
        parts.append(f"Formula pattern {raw['formula_anonymous']}.")
    if domain_matches:
        domains = ", ".join(name.replace("_", " ") for name, _ in domain_matches)
        parts.append(f"Suitable for: {domains}.")
    else:
        parts.append("Matches no application-domain rule.")

    available, missing = [], []
    for extract, label, unit in _OPTIONAL_PROPERTIES:
        value = extract(raw)
        if value is None:
            missing.append(label)
        else:
            available.append(f"{label} {value:.2f}{(' ' + unit) if unit else ''}")
    if available:
        text = "; ".join(available)
        parts.append(f"{text[0].upper()}{text[1:]}.")
    if missing:
        parts.append(f"Not available in Materials Project: {', '.join(missing)}.")
    if elastic_status(raw) == "unphysical":
        parts.append("(Materials Project's elastic tensor for it is unphysical - negative modulus or anisotropy - and was discarded.)")
    return " ".join(parts)


# --- graph construction -----------------------------------------------------


class NodeFactory:
    """Builds DataPoints for materials and the shared nodes they point at.

    Shared nodes are cached per factory, and every shared node type declares
    identity_fields, so a node built here gets the same deterministic id as
    the one already in the graph - which is how a material ingested later
    (matgraph.ingest) attaches to the existing Element/SpaceGroup/... hubs
    instead of duplicating them.
    """

    def __init__(self, config: dict[str, Any], elements_table: dict[str, dict[str, Any]]):
        self.config = config
        self.elements_table = elements_table
        self._rules_by_domain = {r["domain"]: r["rule"] for r in config["application_rules"]}
        self._cache: dict[tuple, Any] = {}

    def _cached(self, key: tuple, build):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    def element(self, symbol: str) -> Element:
        info = self.elements_table[symbol]
        return self._cached(("Element", symbol), lambda: Element(
            symbol=symbol,
            name=info["name"],
            atomic_number=info["atomic_number"],
            group=info["group"],
            period=info["period"],
            electronegativity=info["electronegativity"],
            covalent_radius_pm=info["covalent_radius_pm"],
            category=info["category"],
        ))

    def crystal_system(self, name: str) -> CrystalSystem:
        return self._cached(("CrystalSystem", name), lambda: CrystalSystem(name=name))

    def space_group(self, symbol: str, number: int, crystal_system_name: str) -> SpaceGroup:
        return self._cached(("SpaceGroup", symbol, number), lambda: SpaceGroup(
            symbol=symbol, number=number, crystal_system=self.crystal_system(crystal_system_name),
        ))

    def chemsys(self, chemsys: str) -> ChemicalSystem:
        return self._cached(("ChemicalSystem", chemsys), lambda: ChemicalSystem(chemsys=chemsys))

    def property_class(self, kind: str, name: str, description: str) -> PropertyClass:
        return self._cached(("PropertyClass", kind, name),
                            lambda: PropertyClass(kind=kind, name=name, description=description))

    def domain(self, name: str) -> ApplicationDomain:
        return self._cached(("ApplicationDomain", name), lambda: ApplicationDomain(
            name=name, description=self._rules_by_domain.get(name, name)
        ))

    def oxidation_state(self, species: str) -> OxidationState:
        return self._cached(("OxidationState", species), lambda: OxidationState(species=species))

    def formula_pattern(self, pattern: str) -> FormulaPattern:
        return self._cached(("FormulaPattern", pattern), lambda: FormulaPattern(pattern=pattern))

    def classifications(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        if raw.get("band_gap") is not None:
            out.append(classify_band_gap(raw["band_gap"], self.config))
        if raw.get("energy_above_hull") is not None:
            out.append(classify_stability(raw["energy_above_hull"], self.config))
        ordering_class = classify_ordering(raw.get("ordering"))
        if ordering_class:
            out.append(ordering_class)
        return out

    def material(self, raw: dict[str, Any], origin: str = "mp") -> Material:
        """Material node with every edge except similar_to (needs all materials)."""
        classifications = self.classifications(raw)
        # Domain rules need the full property set; a composition-only record
        # (matgraph.ingest) has none, so no rule can fire for it.
        has_properties = raw.get("band_gap") is not None and raw.get("energy_above_hull") is not None
        domain_matches = apply_application_rules(raw, self.config) if has_properties else []

        symmetry = raw.get("symmetry")
        formula_pattern = raw.get("formula_anonymous")
        return Material(
            material_id=raw["material_id"],
            formula=raw["formula_pretty"],
            description=raw.get("description") or f"{raw['formula_pretty']} (no description available)",
            profile=build_profile(raw, classifications, domain_matches, self.elements_table)
            if has_properties else raw.get("profile", ""),
            origin=origin,
            band_gap=raw.get("band_gap"),
            is_gap_direct=raw.get("is_gap_direct"),
            is_metal=raw.get("is_metal"),
            formation_energy_per_atom=raw.get("formation_energy_per_atom"),
            energy_above_hull=raw.get("energy_above_hull"),
            is_stable=raw.get("is_stable"),
            density=raw.get("density"),
            volume=raw.get("volume"),
            nsites=raw.get("nsites"),
            total_magnetization=raw.get("total_magnetization"),
            ordering=raw.get("ordering"),
            theoretical=raw.get("theoretical"),
            cbm=raw.get("cbm"),
            vbm=raw.get("vbm"),
            efermi=raw.get("efermi"),
            is_magnetic=raw.get("is_magnetic") or False,
            num_magnetic_sites=raw.get("num_magnetic_sites") or 0,
            **(elastic_properties(raw) or {}),
            dielectric_total=raw.get("e_total"),
            dielectric_electronic=raw.get("e_electronic"),
            dielectric_ionic=raw.get("e_ionic"),
            refractive_index=raw.get("n"),
            contains=[self.element(sym) for sym in raw["elements"]],
            has_space_group=(
                (Edge(relationship_type="has_space_group"),
                 self.space_group(symmetry["symbol"], symmetry["number"], symmetry["crystal_system"]))
                if symmetry else None
            ),
            member_of=(Edge(relationship_type="member_of"), self.chemsys(raw["chemsys"])),
            classified_as=[self.property_class(c["kind"], c["name"], c["description"]) for c in classifications],
            suitable_for=[
                (Edge(relationship_type="suitable_for", properties={"rule": rule}), self.domain(name))
                for name, rule in domain_matches
            ],
            has_oxidation_state=[self.oxidation_state(sp) for sp in raw.get("possible_species") or []],
            has_formula_pattern=(
                (Edge(relationship_type="has_formula_pattern"), self.formula_pattern(formula_pattern))
                if formula_pattern else None
            ),
            # similar_to is filled in by the caller once target nodes exist.
        )


def build_graph(
    raw_materials: list[dict[str, Any]],
    config: dict[str, Any],
    elements_table: dict[str, dict[str, Any]],
) -> list[Material]:
    """Build the full set of interconnected DataPoint nodes for all materials."""
    factory = NodeFactory(config, elements_table)
    similarity_edges = compute_similarity_edges(
        raw_materials, config["similarity"]["min_edge_weight"], config["similarity"]["top_k"]
    )
    material_nodes = {raw["material_id"]: factory.material(raw) for raw in raw_materials}

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
