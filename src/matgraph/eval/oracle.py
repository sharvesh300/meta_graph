"""Ground truth for the retrieval benchmark, computed independently of the graph.

Reads the raw Materials Project JSON directly and re-implements every
classification and application rule from its *definition* (the thresholds
and rule text in config/materials.yaml) - it deliberately imports nothing
from matgraph.enrich or matgraph.retriever. If the graph build or the
retriever has a bug, the oracle disagrees with it instead of inheriting it.

What the oracle measures is retrieval *fidelity to the data*, not scientific
correctness of the data or of the rules themselves (see §7 of the plan).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from matgraph.query_schema import QueryConstraints

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
CONFIG_PATH = ROOT / "config" / "materials.yaml"


@dataclass
class OracleMaterial:
    material_id: str
    formula: str
    elements: set[str]
    species: set[str]
    pattern: str | None
    space_group: str
    crystal_system: str
    classes: set[tuple[str, str]]
    domains: set[str]
    numeric: dict[str, float | None]
    flags: dict[str, bool | None]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


def _vrh(value):
    return value.get("vrh") if isinstance(value, dict) else None


def _elastic(raw: dict[str, Any]) -> dict[str, float | None]:
    """Elastic fields, treating a mechanically unstable fit (non-positive
    modulus, negative universal anisotropy) as no data at all."""
    k, g, a = _vrh(raw.get("bulk_modulus")), _vrh(raw.get("shear_modulus")), raw.get("universal_anisotropy")
    valid = k is not None and g is not None and k > 0 and g > 0 and (a is None or a >= 0)
    if not valid:
        return {"bulk_modulus_vrh": None, "shear_modulus_vrh": None,
                "universal_anisotropy": None, "homogeneous_poisson": None}
    return {"bulk_modulus_vrh": k, "shear_modulus_vrh": g,
            "universal_anisotropy": a, "homogeneous_poisson": raw.get("homogeneous_poisson")}


def _bucket(value: float, buckets: list[dict]) -> str:
    for b in buckets:
        if b.get("min", float("-inf")) <= value <= b.get("max", float("inf")):
            return b["name"]
    raise ValueError(value)


def _domains(raw: dict[str, Any], transition_metals: set[str]) -> set[str]:
    """The five rules, re-implemented from their text in materials.yaml."""
    els = set(raw["elements"])
    gap, hull, metal = raw["band_gap"], raw["energy_above_hull"], raw["is_metal"]
    mag, order = abs(raw["total_magnetization"]), raw["ordering"]
    out = set()
    if "Li" in els and els & transition_metals and hull <= 0.05:
        out.add("battery_cathode")
    if gap > 0 and gap <= 3.5 and not metal:
        out.add("semiconductor_device")
    if "O" in els and 1.5 <= gap <= 3.5:
        out.add("photocatalyst")
    if gap > 2 and mag < 0.1:
        out.add("dielectric")
    if order == "AFM" or (order in ("FM", "FiM") and mag >= 0.1):
        out.add("magnetic_material")
    return out


def load_oracle_materials() -> dict[str, OracleMaterial]:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    tms = set(config["transition_metals"])
    ids = json.loads((RAW_DIR / "_manifest.json").read_text())
    out = {}
    for mid in ids:
        raw = json.loads((RAW_DIR / f"{mid}.json").read_text())
        classes = {
            ("band_gap", _bucket(raw["band_gap"], config["band_gap_buckets"])),
            ("stability", _bucket(raw["energy_above_hull"], config["stability_buckets"])),
        }
        if raw.get("ordering") and raw["ordering"] != "Unknown":
            classes.add(("magnetic_ordering", raw["ordering"]))
        numeric = {
            "band_gap": raw["band_gap"],
            "formation_energy_per_atom": raw["formation_energy_per_atom"],
            "energy_above_hull": raw["energy_above_hull"],
            "density": raw["density"],
            "total_magnetization": raw["total_magnetization"],
            "dielectric_total": raw.get("e_total"),
            "dielectric_electronic": raw.get("e_electronic"),
            "dielectric_ionic": raw.get("e_ionic"),
            "refractive_index": raw.get("n"),
            **_elastic(raw),
        }
        out[mid] = OracleMaterial(
            material_id=mid, formula=raw["formula_pretty"], elements=set(raw["elements"]),
            species=set(raw.get("possible_species") or []), pattern=raw.get("formula_anonymous"),
            space_group=raw["symmetry"]["symbol"], crystal_system=raw["symmetry"]["crystal_system"],
            classes=classes, domains=_domains(raw, tms), numeric=numeric,
            flags={"is_metal": raw["is_metal"], "is_gap_direct": raw["is_gap_direct"],
                   "is_stable": raw["is_stable"], "theoretical": raw["theoretical"],
                   "is_magnetic": raw.get("is_magnetic")},
            raw=raw,
        )
    return out


def satisfied_fraction(m: OracleMaterial, c: QueryConstraints) -> float:
    """Share of the query's hard constraints `m` satisfies (1.0 = all).
    Missing numeric data counts as not satisfied."""
    checks: list[bool] = []
    checks += [d in m.domains for d in c.application_domains]
    checks += [(pc.kind, pc.name) in m.classes for pc in c.property_classes]
    checks += [e in m.elements for e in c.include_elements]
    checks += [s in m.species for s in c.oxidation_states]
    if c.exclude_elements:
        checks.append(not (m.elements & set(c.exclude_elements)))
    if c.crystal_systems:
        checks.append(m.crystal_system in c.crystal_systems)
    if c.space_groups:
        checks.append(m.space_group in c.space_groups)
    if c.formula_patterns:
        checks.append(m.pattern in c.formula_patterns)
    for r in c.property_ranges:
        v = m.numeric.get(r.property)
        checks.append(v is not None and (r.min is None or v >= r.min) and (r.max is None or v <= r.max))
    for b in c.boolean_flags:
        checks.append(m.flags.get(b.flag) is not None and bool(m.flags[b.flag]) == b.value)
    if c.sort_by:
        checks.append(m.numeric.get(c.sort_by) is not None)
    return sum(checks) / len(checks) if checks else 1.0


def relevance(
    materials: dict[str, OracleMaterial], c: QueryConstraints
) -> dict[str, float]:
    """Graded relevance per material_id (only non-zero entries returned).

    - Filter queries: binary - 1.0 if every hard constraint holds, else not
      relevant. (An earlier version gave near-misses partial grades; that put
      constraint-violating materials into the *ideal* ranking, so a method
      returning exactly the 2 true answers scored below one padding its list
      with violations. Near-misses are measured by the violation rate instead.)
    - Anchor queries: must satisfy the hard constraints AND share the
      anchor's formula_anonymous (grade 0.5), +0.5 for the same crystal system.
    - sort_by queries: the true top_k by that property are relevant, graded by rank.
    """
    by_formula = {m.formula: m for m in materials.values()}
    anchors = [by_formula.get(a) or materials.get(a) for a in c.anchor_materials]
    anchors = [a for a in anchors if a]
    anchor_ids = {a.material_id for a in anchors}

    if c.sort_by and not (c.property_ranges or c.application_domains or c.include_elements):
        ranked = sorted(
            (m for m in materials.values() if m.numeric.get(c.sort_by) is not None),
            key=lambda m: m.numeric[c.sort_by], reverse=c.sort_descending,
        )[: c.top_k]
        return {m.material_id: 1.0 - i / (2 * c.top_k) for i, m in enumerate(ranked)}

    grades: dict[str, float] = {}
    for mid, m in materials.items():
        if mid in anchor_ids:
            continue
        frac = satisfied_fraction(m, c)
        if anchors:
            if frac < 1.0:
                continue
            best = 0.0
            for a in anchors:
                if m.pattern == a.pattern:
                    best = max(best, 0.5 + (0.5 if m.crystal_system == a.crystal_system else 0.0))
            if best:
                grades[mid] = best
        elif frac == 1.0:
            grades[mid] = 1.0
    return grades


def is_hit(grade: float, anchored: bool) -> bool:
    """Binary relevance used by Recall@k / MRR."""
    return grade >= (0.5 if anchored else 1.0)
