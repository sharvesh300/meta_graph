"""Typed query constraints: the contract between query parsing and retrieval.

The LLM (or the rule-based fallback) turns a natural-language question into a
``QueryConstraints``; retrieval then enforces it deterministically against
the graph. Enumerated fields use ``Literal`` so the LLM's JSON schema lists
the allowed values explicitly; free-form fields (elements, space groups,
materials, ...) are grounded against the graph vocabulary afterwards
(``query_parser.ground_constraints``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Numeric Material attributes a query may constrain: name -> (unit, meaning).
# Coverage notes matter: elastic/dielectric fields exist for ~18-20% of
# materials, so retrieval reports how many candidates lacked the value.
NUMERIC_PROPERTIES: dict[str, tuple[str, str]] = {
    "band_gap": ("eV", "DFT band gap (systematically underestimates experiment)"),
    "formation_energy_per_atom": ("eV/atom", "formation energy per atom"),
    "energy_above_hull": ("eV/atom", "energy above the convex hull; 0 = thermodynamically stable"),
    "density": ("g/cm3", "mass density"),
    "total_magnetization": ("bohr magnetons", "total magnetization of the cell"),
    "bulk_modulus_vrh": ("GPa", "bulk modulus, Voigt-Reuss-Hill average (partial coverage)"),
    "shear_modulus_vrh": ("GPa", "shear modulus, Voigt-Reuss-Hill average (partial coverage)"),
    "universal_anisotropy": ("", "elastic universal anisotropy index (partial coverage)"),
    "homogeneous_poisson": ("", "Poisson's ratio (partial coverage)"),
    "dielectric_total": ("", "total static dielectric constant (partial coverage)"),
    "dielectric_electronic": ("", "electronic dielectric constant (partial coverage)"),
    "dielectric_ionic": ("", "ionic dielectric constant (partial coverage)"),
    "refractive_index": ("", "refractive index (partial coverage)"),
}

NumericProperty = Literal[
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
    "dielectric_electronic",
    "dielectric_ionic",
    "refractive_index",
]

BooleanFlag = Literal["is_metal", "is_gap_direct", "is_stable", "theoretical", "is_magnetic"]

ApplicationDomainName = Literal[
    "battery_cathode", "semiconductor_device", "photocatalyst", "dielectric", "magnetic_material"
]

PropertyClassKind = Literal["band_gap", "stability", "magnetic_ordering"]

CrystalSystemName = Literal[
    "Cubic", "Hexagonal", "Trigonal", "Tetragonal", "Orthorhombic", "Monoclinic", "Triclinic"
]


class PropertyRange(BaseModel):
    property: NumericProperty
    min: float | None = Field(default=None, description="inclusive lower bound, null if none")
    max: float | None = Field(default=None, description="inclusive upper bound, null if none")


class BooleanConstraint(BaseModel):
    flag: BooleanFlag
    value: bool


class PropertyClassRef(BaseModel):
    kind: PropertyClassKind
    name: str = Field(
        description="band_gap: metal|narrow_gap|semiconductor|wide_gap; "
        "stability: stable|metastable|unstable; magnetic_ordering: FM|AFM|FiM|NM"
    )


class QueryConstraints(BaseModel):
    """Everything a question can require of a material. Semantics:

    - list-valued node constraints are ALL-of for include_elements /
      oxidation_states / application_domains / property_classes, and ANY-of
      for crystal_systems / space_groups / formula_patterns (a material has
      exactly one of each);
    - anchor_materials make it a similarity query ("materials like X").
    """

    application_domains: list[ApplicationDomainName] = Field(default_factory=list)
    property_ranges: list[PropertyRange] = Field(default_factory=list)
    boolean_flags: list[BooleanConstraint] = Field(default_factory=list)
    property_classes: list[PropertyClassRef] = Field(default_factory=list)
    include_elements: list[str] = Field(
        default_factory=list, description="element symbols the material must contain"
    )
    exclude_elements: list[str] = Field(
        default_factory=list, description="element symbols the material must NOT contain"
    )
    crystal_systems: list[CrystalSystemName] = Field(default_factory=list)
    space_groups: list[str] = Field(default_factory=list, description="e.g. Fm-3m, P6_3mc")
    formula_patterns: list[str] = Field(
        default_factory=list, description="anonymized formula, e.g. ABC3 (perovskite), AB2C4 (spinel)"
    )
    oxidation_states: list[str] = Field(default_factory=list, description="e.g. Ti4+, O2-, Fe2+")
    anchor_materials: list[str] = Field(
        default_factory=list,
        description="formulas or material_ids the user wants similar materials to",
    )
    semantic_text: str = Field(
        default="",
        description="descriptive wording not captured by the fields above "
        "(e.g. 'layered', 'octahedral coordination'); empty if none",
    )
    top_k: int = Field(default=10, ge=1, le=50)
    sort_by: NumericProperty | None = Field(
        default=None, description="only if the user asks for highest/lowest of a property"
    )
    sort_descending: bool = True

    def is_empty(self) -> bool:
        return not any(
            [
                self.application_domains, self.property_ranges, self.boolean_flags,
                self.property_classes, self.include_elements, self.exclude_elements,
                self.crystal_systems, self.space_groups, self.formula_patterns,
                self.oxidation_states, self.anchor_materials, self.semantic_text.strip(),
            ]
        )


class IgnoredConstraint(BaseModel):
    field: str
    value: str
    reason: str


class ParsedQuery(BaseModel):
    """A question after parsing and vocabulary grounding."""

    question: str
    constraints: QueryConstraints
    ignored_constraints: list[IgnoredConstraint] = Field(default_factory=list)
    parser: Literal["llm", "llm+rules", "rules", "gold"] = "rules"
    parser_model: str | None = None
