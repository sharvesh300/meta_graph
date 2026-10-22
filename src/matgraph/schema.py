"""Cognee DataPoint schema for the materials knowledge graph.

Design rule: continuous numerics stay as attributes on Material (exact values
for filtering/comparison); anything used for grouping or traversal becomes a
node; derived similarity becomes a weighted edge.

Every node type that multiple materials can share (Element, CrystalSystem,
SpaceGroup, ChemicalSystem, PropertyClass, ApplicationDomain, OxidationState,
FormulaPattern) declares ``identity_fields`` so repeated runs and repeated
references deduplicate onto the same graph node instead of creating copies.
"""

from __future__ import annotations

from typing import Any

from pydantic import SkipValidation

from cognee.low_level import DataPoint
from cognee.infrastructure.engine.models.Edge import Edge  # noqa: F401  (re-exported for builders)


class CrystalSystem(DataPoint):
    name: str  # e.g. "Cubic", "Hexagonal", "Orthorhombic"
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class SpaceGroup(DataPoint):
    symbol: str  # e.g. "Fd-3m"
    number: int
    crystal_system: CrystalSystem
    metadata: dict = {"index_fields": ["symbol"], "identity_fields": ["symbol", "number"]}


class ChemicalSystem(DataPoint):
    chemsys: str  # e.g. "Fe-Li-O-P"
    metadata: dict = {"index_fields": ["chemsys"], "identity_fields": ["chemsys"]}


class Element(DataPoint):
    symbol: str
    name: str
    atomic_number: int
    group: int
    period: int
    electronegativity: float
    covalent_radius_pm: float
    category: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["symbol"]}


class PropertyClass(DataPoint):
    """A categorical bucket a material can be classified into.

    kind distinguishes independent bucket families (band_gap, stability,
    magnetic_ordering) so e.g. "stable" (stability) and "wide_gap" (band_gap)
    are both reachable without collision.
    """

    kind: str  # "band_gap" | "stability" | "magnetic_ordering"
    name: str  # e.g. "wide_gap", "stable", "AFM"
    description: str
    metadata: dict = {
        "index_fields": ["description"],
        "identity_fields": ["kind", "name"],
    }


class ApplicationDomain(DataPoint):
    name: str  # e.g. "battery_cathode"
    description: str
    metadata: dict = {"index_fields": ["description"], "identity_fields": ["name"]}


class OxidationState(DataPoint):
    """An ionic species (e.g. "Fe2+", "O2-"). Shared across every material that
    contains it, so materials with the same oxidation state of an element
    become connected - this is also the anchor Phase 3's ionic-substitution
    model will reason over.
    """

    species: str  # e.g. "Fe2+"
    metadata: dict = {"index_fields": ["species"], "identity_fields": ["species"]}


class FormulaPattern(DataPoint):
    """Anonymized stoichiometric pattern (e.g. "ABC3" for perovskites, "AB2O4"
    for spinels), from Materials Project's formula_anonymous. Groups materials
    by stoichiometric shape independent of actual composition.
    """

    pattern: str
    metadata: dict = {"index_fields": ["pattern"], "identity_fields": ["pattern"]}


class Material(DataPoint):
    material_id: str
    formula: str
    description: str  # robocrystallographer text, embeddable
    # Deterministic property summary (enrich.build_profile), embeddable. The
    # robocrys description covers structure only; this gives semantic search
    # and the Vector RAG baseline the property/class/domain facts too.
    profile: str = ""
    # Where this node came from: "mp" for the curated cluster build, "user"
    # for materials added later through matgraph.ingest.
    origin: str = "mp"
    # Where the property values came from: "mp" (Materials Project record),
    # "composition" (formula only) or "cgcnn" (predicted from a user-supplied
    # structure; material_id "struct-...").
    property_source: str = "mp"

    # Core computed properties. Always present for Materials Project records;
    # None only for composition-only materials added through matgraph.ingest
    # (a formula Materials Project has no entry for - nothing to report).
    band_gap: float | None = None
    is_gap_direct: bool | None = None
    is_metal: bool | None = None
    formation_energy_per_atom: float | None = None
    energy_above_hull: float | None = None
    is_stable: bool | None = None
    density: float | None = None
    volume: float | None = None
    nsites: int | None = None
    total_magnetization: float | None = None
    ordering: str | None = None
    theoretical: bool | None = None

    # Electronic structure (cheap scalars, always paired with band_gap).
    cbm: float | None = None
    vbm: float | None = None
    efermi: float | None = None
    is_magnetic: bool = False
    num_magnetic_sites: int = 0

    # Elastic properties (PRD: "Elastic Properties"). Not computed for every
    # material - None when Materials Project has no elastic tensor for it.
    bulk_modulus_vrh: float | None = None
    shear_modulus_vrh: float | None = None
    universal_anisotropy: float | None = None
    homogeneous_poisson: float | None = None

    # Dielectric properties (PRD: "Dielectric Properties"). Also partial coverage.
    dielectric_total: float | None = None
    dielectric_electronic: float | None = None
    dielectric_ionic: float | None = None
    refractive_index: float | None = None

    # CGCNN predictions (Phase 3, matgraph.cgcnn.interpret). Never written into
    # the DFT fields above: on a struct- node those stay None. prediction_model
    # names the pinned model set; expected errors per attribute are in
    # artifacts/eval/cgcnn_graph_eval.json.
    predicted_formation_energy_per_atom: float | None = None
    predicted_band_gap: float | None = None
    predicted_is_metal: bool | None = None
    predicted_metal_probability: float | None = None
    predicted_efermi: float | None = None
    predicted_bulk_modulus_vrh: float | None = None
    predicted_shear_modulus_vrh: float | None = None
    predicted_homogeneous_poisson: float | None = None
    prediction_model: str | None = None

    # These are relationship fields: builders pass a (Edge(...), target(s)) tuple
    # as the value (see build_material in enrich.py). SkipValidation is required
    # because pydantic cannot type-check a tuple-of-(Edge, DataPoint) shape.
    contains: SkipValidation[Any] = None  # (Edge, list[Element])
    has_space_group: SkipValidation[Any] = None  # (Edge, SpaceGroup)
    member_of: SkipValidation[Any] = None  # (Edge, ChemicalSystem)
    classified_as: SkipValidation[Any] = None  # (Edge, list[PropertyClass])
    suitable_for: SkipValidation[Any] = None  # (Edge, list[ApplicationDomain])
    similar_to: SkipValidation[Any] = None  # (Edge, list[Material])
    has_oxidation_state: SkipValidation[Any] = None  # (Edge, list[OxidationState])
    has_formula_pattern: SkipValidation[Any] = None  # (Edge, FormulaPattern)

    metadata: dict = {
        "index_fields": ["formula", "description", "profile"],
        "identity_fields": ["material_id"],
    }
