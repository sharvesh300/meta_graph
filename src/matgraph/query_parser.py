"""Natural-language question -> grounded QueryConstraints.

Two parsers fill the same schema:

- ``parse_with_llm``: the LLM with JSON-schema-constrained output (primary).
- ``parse_with_rules``: deterministic regex/synonym parser. Used in
  ``--no-llm`` mode, in tests, and as the fallback when OpenRouter is
  unavailable. Deliberately conservative: it would rather miss a constraint
  (it lands in the semantic text) than invent one.

Either way, ``ground_constraints`` then checks every value against the
graph's actual vocabulary. Anything that does not map is reported in
``ignored_constraints`` - never silently dropped, never passed to retrieval.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from matgraph.enrich import load_config, load_elements_table
from matgraph.query_schema import (
    NUMERIC_PROPERTIES,
    BooleanConstraint,
    IgnoredConstraint,
    ParsedQuery,
    PropertyClassRef,
    PropertyRange,
    QueryConstraints,
)

if TYPE_CHECKING:
    from matgraph.graph_store import Graph

logger = logging.getLogger(__name__)

# --- shared vocabulary ------------------------------------------------------

DOMAIN_SYNONYMS: list[tuple[str, str]] = [
    (r"\b(battery|batteries|cathodes?|li-?ion|lithium-?ion|intercalation)\b", "battery_cathode"),
    (r"\b(photocataly\w*|water[- ]splitting|solar[- ]fuel)\b", "photocatalyst"),
    (r"\b(dielectrics?(?! constant)|capacitors?|insulators?|insulating)\b", "dielectric"),
    (r"\b(magnets?|magnetic materials?|spintronic\w*)\b", "magnetic_material"),
    (r"\b(semiconductors?|semiconducting|semiconductor devices?|transistors?|optoelectronic\w*|leds?)\b",
     "semiconductor_device"),
]

ORDERING_WORDS: list[tuple[str, str]] = [  # matched case-insensitively
    (r"\bantiferromagnet\w*", "AFM"),
    (r"\bferrimagnet\w*", "FiM"),
    (r"\bferromagnet\w*", "FM"),
    (r"\bnon-?magnetic\b", "NM"),
]
ORDERING_ABBREVIATIONS = ["AFM", "FiM", "FM", "NM"]  # matched case-sensitively

STABILITY_WORDS: list[tuple[str, str]] = [
    (r"\bmetastable\b", "metastable"),
    (r"\bunstable\b", "unstable"),
    (r"\bstable\b", "stable"),
]

BAND_GAP_CLASS_WORDS: list[tuple[str, str]] = [
    (r"\bwide[- ](band[- ]?)?gap\b|\bultra-?wide[- ]gap\b", "wide_gap"),
    (r"\bnarrow[- ](band[- ]?)?gap\b|\bsmall[- ]gap\b", "narrow_gap"),
    (r"\bmetals?\b|\bmetallic\b|\bconductors?\b|\bconducting\b", "metal"),
]

CRYSTAL_SYSTEMS = ["Cubic", "Hexagonal", "Trigonal", "Tetragonal", "Orthorhombic", "Monoclinic", "Triclinic"]
CRYSTAL_SYSTEM_SYNONYMS = {"rhombohedral": "Trigonal"}

# Structure-family words -> anonymized formula pattern.
FORMULA_PATTERN_WORDS: list[tuple[str, str]] = [
    (r"\bperovskites?\b", "ABC3"),
    (r"\bspinels?\b", "AB2C4"),
    (r"\bolivines?\b", "ABCD4"),
]

# Compound-class nouns -> element that must be present.
ANION_CLASS_WORDS: list[tuple[str, str]] = [
    (r"\boxides?\b", "O"),
    (r"\bnitrides?\b", "N"),
    (r"\bsulfides?\b|\bsulphides?\b", "S"),
    (r"\bselenides?\b", "Se"),
    (r"\btellurides?\b", "Te"),
    (r"\bphosphides?\b", "P"),
    (r"\bphosphates?\b", "P"),
    (r"\barsenides?\b", "As"),
    (r"\bantimonides?\b", "Sb"),
    (r"\bfluorides?\b", "F"),
    (r"\bcarbides?\b", "C"),
]

# Natural-language names for numeric properties (longest first matters).
PROPERTY_SYNONYMS: list[tuple[str, str]] = [
    (r"energy above (the )?(convex )?hull|e_?hull|hull energy", "energy_above_hull"),
    (r"formation energy|formation enthalpy", "formation_energy_per_atom"),
    (r"band ?gap|bandgap|gap", "band_gap"),
    (r"bulk modulus", "bulk_modulus_vrh"),
    (r"shear modulus", "shear_modulus_vrh"),
    (r"poisson'?s? ratio", "homogeneous_poisson"),
    (r"(elastic )?anisotropy", "universal_anisotropy"),
    (r"electronic dielectric constant", "dielectric_electronic"),
    (r"ionic dielectric constant", "dielectric_ionic"),
    (r"dielectric constant|permittivity", "dielectric_total"),
    (r"refractive index", "refractive_index"),
    (r"density", "density"),
    (r"(total )?magneti[sz]ation|magnetic moment", "total_magnetization"),
]

# Two-letter symbols that are also everyday words; only accepted as elements
# in explicit contexts ("containing In", "In-based").
_AMBIGUOUS_SYMBOLS = {"In", "As", "Be", "At", "No", "He", "Am", "Es"}

_NUM = r"(-?\d+(?:\.\d+)?)"
_ABOVE = r"(?:above|greater than|more than|over|at least|exceeding|higher than|>=?|≥|≳|min(?:imum)? of)"
_BELOW = r"(?:below|less than|under|at most|lower than|up to|<=?|≤|≲|max(?:imum)? of|no more than)"
_BETWEEN = rf"(?:between|from)\s+{_NUM}\s*(?:eV(?:/atom)?|GPa|g/cm3)?\s*(?:and|to|-|–)\s*{_NUM}"
_FORMULA = r"(?:[A-Z][a-z]?\d*(?:\.\d+)?){1,6}"


def _elements_by_name() -> dict[str, str]:
    table = load_elements_table()
    names = {info["name"].lower(): sym for sym, info in table.items()}
    names["sulphur"] = "S"
    names["aluminum"] = "Al"
    names["caesium"] = "Cs"
    return names


# --- rule-based parser --------------------------------------------------------


def _add_range(ranges: dict[str, PropertyRange], prop: str, lo: float | None, hi: float | None) -> None:
    current = ranges.get(prop) or PropertyRange(property=prop)
    if lo is not None:
        current.min = lo if current.min is None else max(current.min, lo)
    if hi is not None:
        current.max = hi if current.max is None else min(current.max, hi)
    ranges[prop] = current


def parse_with_rules(question: str) -> QueryConstraints:
    q = question.strip()
    ql = q.lower()
    c = QueryConstraints()

    # Numeric ranges: "<property> <comparator> <number>" and "<property> between X and Y".
    ranges: dict[str, PropertyRange] = {}
    for pattern, prop in PROPERTY_SYNONYMS:
        for m in re.finditer(rf"(?:{pattern})(?:\s+(?:of|is|=|:))?\s*{_BETWEEN}", ql):
            lo, hi = float(m.group(m.lastindex - 1)), float(m.group(m.lastindex))
            _add_range(ranges, prop, min(lo, hi), max(lo, hi))
        for m in re.finditer(rf"(?:{pattern})(?:\s+(?:of|is))?\s*{_ABOVE}\s*{_NUM}", ql):
            _add_range(ranges, prop, float(m.group(m.lastindex)), None)
        for m in re.finditer(rf"(?:{pattern})(?:\s+(?:of|is))?\s*{_BELOW}\s*{_NUM}", ql):
            _add_range(ranges, prop, None, float(m.group(m.lastindex)))
        ql = re.sub(rf"(?:{pattern})", " ", ql) if prop in ranges else ql
    c.property_ranges = list(ranges.values())

    ql = question.lower()

    # Application domains.
    for pattern, domain in DOMAIN_SYNONYMS:
        if re.search(pattern, ql) and domain not in c.application_domains:
            c.application_domains.append(domain)

    # Magnetic ordering (checked case-sensitively for the abbreviations).
    ordering = next((o for pattern, o in ORDERING_WORDS if re.search(pattern, ql)), None)
    ordering = ordering or next((a for a in ORDERING_ABBREVIATIONS if re.search(rf"\b{a}\b", q)), None)
    if ordering:
        c.property_classes.append(PropertyClassRef(kind="magnetic_ordering", name=ordering))
    if any(pc.kind == "magnetic_ordering" for pc in c.property_classes):
        # "ferromagnetic" etc. already specifies magnetism; don't also demand the domain rule.
        c.application_domains = [d for d in c.application_domains if d != "magnetic_material"]

    # Stability and band-gap classes.
    for pattern, name in STABILITY_WORDS:
        if re.search(pattern, ql) and not re.search(r"\bmost stable\b", ql):
            c.property_classes.append(PropertyClassRef(kind="stability", name=name))
            break
    for pattern, name in BAND_GAP_CLASS_WORDS:
        if re.search(pattern, ql) and not re.search(r"\bnon-?metal", ql):
            if name == "metal" and re.search(r"transition metals?|alkali metals?|metal (ions?|cations?|oxides?)", ql):
                continue
            c.property_classes.append(PropertyClassRef(kind="band_gap", name=name))
            break
    if re.search(r"\bdirect[- ](band[- ]?)?gap\b", ql) and not re.search(r"\bindirect", ql):
        c.boolean_flags.append(BooleanConstraint(flag="is_gap_direct", value=True))
    elif re.search(r"\bindirect[- ](band[- ]?)?gap\b", ql):
        c.boolean_flags.append(BooleanConstraint(flag="is_gap_direct", value=False))
    if re.search(r"\bnon-?metallic\b|\bnon-?metals?\b", ql):
        c.boolean_flags.append(BooleanConstraint(flag="is_metal", value=False))
    if re.search(r"\bexperimentally (known|observed|synthesi[sz]ed)\b|\bsynthesi[sz]ed\b", ql):
        c.boolean_flags.append(BooleanConstraint(flag="theoretical", value=False))

    # Crystal systems.
    for cs in CRYSTAL_SYSTEMS:
        if re.search(rf"\b{cs.lower()}\b", ql):
            c.crystal_systems.append(cs)
    for word, cs in CRYSTAL_SYSTEM_SYNONYMS.items():
        if re.search(rf"\b{word}\b", ql) and cs not in c.crystal_systems:
            c.crystal_systems.append(cs)

    # Formula patterns: structure-family words and literal patterns like "AB2C4".
    for pattern, formula_pattern in FORMULA_PATTERN_WORDS:
        if re.search(pattern, ql):
            c.formula_patterns.append(formula_pattern)
    for m in re.finditer(r"\b(A(?:\d*)B\d*(?:C\d*)?(?:D\d*)?)\b", q):
        if m.group(1) not in c.formula_patterns and len(m.group(1)) >= 2:
            c.formula_patterns.append(m.group(1))

    # Space groups by Hermann-Mauguin symbol after "space group" (validated in grounding).
    for m in re.finditer(r"space group\s+([A-Z][A-Za-z0-9_\-/]*)", q):
        c.space_groups.append(m.group(1).rstrip(".,"))

    # Oxidation states: "Ti4+", "O2-", "Li+".
    for m in re.finditer(r"\b([A-Z][a-z]?)(\d?[+-])(?![\w])", q):
        c.oxidation_states.append(m.group(1) + m.group(2))

    # Anchors: "similar to X", "like X", "analogues of X".
    anchor_spans = []
    for m in re.finditer(rf"(?i:similar to|like|analogu?es? (?:of|to)|comparable to|resembl\w*)\s+({_FORMULA}|mp-\w+)", q):
        c.anchor_materials.append(m.group(1))
        anchor_spans.append(m.span(1))

    # Elements: exclusions first ("Co-free", "without cobalt", "no Pb"), then inclusions.
    by_name = _elements_by_name()
    symbols = set(load_elements_table())
    excluded: list[str] = []

    def to_symbol(token: str) -> str | None:
        if token in symbols:
            return token
        return by_name.get(token.lower())

    for m in re.finditer(r"\bfree of\s+([A-Za-z]+)", q):
        sym = to_symbol(m.group(1)) or to_symbol(m.group(1).capitalize())
        if sym:
            excluded.append(sym)
    for m in re.finditer(r"\b([A-Za-z]+)[- ]free\b", q):
        sym = to_symbol(m.group(1)) or to_symbol(m.group(1).capitalize())
        if sym:
            excluded.append(sym)
    for m in re.finditer(r"\b(?:without|excluding|no|not containing|(?:do|does|don't|doesn't)(?: not)? contain|avoid(?:ing)?|except)\s+((?:[A-Za-z]+(?:\s*,\s*|\s+(?:or|and|nor)\s+)?)+)", q):
        for token in re.split(r"\s*,\s*|\s+(?:or|and|nor)\s+", m.group(1)):
            sym = to_symbol(token.strip()) or to_symbol(token.strip().capitalize())
            if sym and sym not in excluded:
                excluded.append(sym)
    c.exclude_elements = excluded

    included: list[str] = []
    for name, sym in by_name.items():
        if re.search(rf"\b{name}\b", ql) and sym not in excluded:
            # "lithium" in "lithium-ion" is a domain word, not a composition requirement.
            if name == "lithium" and re.search(r"lithium-?ion", ql) and not re.search(r"\blithium\b(?!-?ion)", ql):
                continue
            included.append(sym)
    for m in re.finditer(r"\b(?:contain(?:s|ing)?|with|including|based on|based)\s+([A-Z][a-z]?)(?![a-z])(?!\d*[+-])", q):
        if m.group(1) in symbols and m.group(1) not in excluded and m.group(1) not in included:
            included.append(m.group(1))
    for m in re.finditer(r"\b([A-Z][a-z]?)-(?:based|containing|rich|bearing)\b", q):
        if m.group(1) in symbols and m.group(1) not in excluded and m.group(1) not in included:
            included.append(m.group(1))
    # Bare two-letter symbols ("Li cathode", "Fe and Mn") - skipping ones that
    # are also common English words, and "Li-ion" (a domain word).
    for m in re.finditer(r"\b([A-Z][a-z])\b(?!-?ion)(?![+-]?\d*[+-])", q):
        sym = m.group(1)
        if (sym in symbols and sym not in _AMBIGUOUS_SYMBOLS and sym not in excluded
                and sym not in included):
            included.append(sym)
    for pattern, sym in ANION_CLASS_WORDS:
        if re.search(pattern, ql) and sym not in included and sym not in excluded:
            included.append(sym)
    if "ABCD4" in c.formula_patterns or re.search(r"\bphosphates?\b", ql):
        if "O" not in included and "O" not in excluded:
            included.append("O")
    c.include_elements = included

    # Result size and ordering.
    m = re.search(r"\btop\s+(\d+)\b|\b(\d+)\s+(?:materials|candidates|compounds|results)\b", ql)
    if m:
        c.top_k = max(1, min(50, int(m.group(1) or m.group(2))))
    m = re.search(r"\b(highest|largest|biggest|maximum|lowest|smallest|minimum)\s+([a-z' ]+)", ql)
    if m:
        for pattern, prop in PROPERTY_SYNONYMS:
            if re.match(pattern, m.group(2).strip()):
                c.sort_by = prop
                c.sort_descending = m.group(1) in {"highest", "largest", "biggest", "maximum"}
                break
    if re.search(r"\bmost stable\b", ql):
        c.sort_by, c.sort_descending = "energy_above_hull", False

    c.semantic_text = q
    return c


# --- LLM parser ---------------------------------------------------------------

_SYSTEM_PROMPT = """You convert a materials-science question into search constraints \
for a knowledge graph built from the Materials Project. Fill only what the \
question actually asks for; leave everything else empty/null. Never invent \
constraints.

Numeric properties you may constrain (property: unit - meaning):
{properties}

Application domains and the exact rule that defines each:
{domains}

Property classes:
- band_gap: metal (=0 eV), narrow_gap (0-1 eV), semiconductor (1-3 eV), wide_gap (>3 eV)
- stability: stable (on hull), metastable (<=0.05 eV/atom), unstable (>0.05 eV/atom)
- magnetic_ordering: FM, AFM, FiM, NM

Guidance:
- Elements must be chemical symbols (e.g. "lithium" -> "Li"). "X-free" or \
"without X" goes in exclude_elements. "oxides" means include O, "nitrides" N, etc.
- "stable" -> property_classes stability/stable. "wide band gap" -> band_gap/wide_gap. \
An explicit number ("band gap above 2 eV") -> property_ranges instead.
- "perovskite" -> formula_patterns ["ABC3"], "spinel" -> ["AB2C4"], "olivine" -> ["ABCD4"].
- "similar to / like <material>" -> anchor_materials.
- semantic_text: only descriptive words not captured elsewhere (e.g. "layered"); else "".
- sort_by only for superlatives ("highest band gap").
Return JSON only."""


def _system_prompt() -> str:
    config = load_config()
    properties = "\n".join(
        f"- {name}: {unit or 'dimensionless'} - {meaning}"
        for name, (unit, meaning) in NUMERIC_PROPERTIES.items()
    )
    domains = "\n".join(f"- {r['domain']}: {r['rule']}" for r in config["application_rules"])
    return _SYSTEM_PROMPT.format(properties=properties, domains=domains)


def parse_with_llm(question: str) -> tuple[QueryConstraints, str]:
    """Returns (constraints, model_id). Raises llm.LLMUnavailable on failure."""
    from matgraph.llm import complete_json

    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": question},
    ]
    constraints, meta = complete_json(messages, QueryConstraints)
    return constraints, meta["model"]


# --- grounding ----------------------------------------------------------------


def _normalize_species(value: str) -> str:
    """'Ti4+' / 'Ti+4' / 'ti4+' -> 'Ti4+'; 'Li1+' -> 'Li+'."""
    m = re.fullmatch(r"\s*([A-Za-z]{1,2})\s*(?:(\d*)([+-])|([+-])(\d*))\s*", value)
    if not m:
        return value.strip()
    symbol = m.group(1).capitalize()
    digits = m.group(2) if m.group(2) is not None else m.group(5)
    sign = m.group(3) or m.group(4)
    digits = "" if digits in ("", "1") else digits
    return f"{symbol}{digits}{sign}"


def ground_constraints(
    constraints: QueryConstraints, graph: Graph
) -> tuple[QueryConstraints, list[IgnoredConstraint]]:
    """Map every free-form value onto graph vocabulary; report what doesn't map."""
    ignored: list[IgnoredConstraint] = []
    by_name = _elements_by_name()
    known_symbols = {p["symbol"] for _, p in graph.nodes_of_type("Element")}
    table_symbols = set(load_elements_table())

    def element(value: str, field: str) -> str | None:
        v = value.strip()
        sym = v if v in table_symbols else (v.capitalize() if v.capitalize() in table_symbols else by_name.get(v.lower()))
        if not sym:
            ignored.append(IgnoredConstraint(field=field, value=value, reason="not a chemical element"))
            return None
        if field == "include_elements" and sym not in known_symbols:
            # A valid element that no material in the graph contains: keep it
            # (it correctly yields zero results) but tell the user why.
            ignored.append(
                IgnoredConstraint(
                    field=field, value=value,
                    reason=f"{sym} is a valid element but no material in the graph contains it",
                )
            )
        return sym

    g = constraints.model_copy(deep=True)
    g.include_elements = list(dict.fromkeys(s for s in (element(v, "include_elements") for v in g.include_elements) if s))
    g.exclude_elements = list(dict.fromkeys(s for s in (element(v, "exclude_elements") for v in g.exclude_elements) if s))
    both = set(g.include_elements) & set(g.exclude_elements)
    for sym in both:
        ignored.append(IgnoredConstraint(field="include_elements", value=sym, reason="also excluded; exclusion wins"))
    g.include_elements = [s for s in g.include_elements if s not in both]

    valid_classes = {(p["kind"], p["name"]) for _, p in graph.nodes_of_type("PropertyClass")}
    kept_classes = []
    for pc in g.property_classes:
        if (pc.kind, pc.name) in valid_classes:
            kept_classes.append(pc)
        else:
            ignored.append(IgnoredConstraint(field="property_classes", value=f"{pc.kind}:{pc.name}",
                                             reason="no such property class in the graph"))
    g.property_classes = list({(pc.kind, pc.name): pc for pc in kept_classes}.values())

    space_groups = {p["symbol"] for _, p in graph.nodes_of_type("SpaceGroup")}
    kept = []
    for sg in g.space_groups:
        if sg in space_groups:
            kept.append(sg)
        else:
            ignored.append(IgnoredConstraint(field="space_groups", value=sg, reason="no material in the graph has this space group"))
    g.space_groups = kept

    patterns = {p["pattern"] for _, p in graph.nodes_of_type("FormulaPattern")}
    kept = []
    for fp in g.formula_patterns:
        if fp in patterns:
            kept.append(fp)
        else:
            ignored.append(IgnoredConstraint(field="formula_patterns", value=fp, reason="no material in the graph has this formula pattern"))
    g.formula_patterns = kept

    species = {p["species"] for _, p in graph.nodes_of_type("OxidationState")}
    kept = []
    for sp in g.oxidation_states:
        norm = _normalize_species(sp)
        if norm in species:
            kept.append(norm)
        else:
            ignored.append(IgnoredConstraint(field="oxidation_states", value=sp, reason="no material in the graph has this oxidation state"))
    g.oxidation_states = kept

    kept = []
    for anchor in g.anchor_materials:
        if graph.material(anchor.strip()):
            kept.append(anchor.strip())
        else:
            ignored.append(IgnoredConstraint(
                field="anchor_materials", value=anchor,
                reason="not in the graph (add it with scripts/add_material.py to use it as an anchor)",
            ))
    g.anchor_materials = kept

    kept_ranges = []
    for r in g.property_ranges:
        if r.min is None and r.max is None:
            ignored.append(IgnoredConstraint(field="property_ranges", value=r.property, reason="range has no bounds"))
        elif r.min is not None and r.max is not None and r.min > r.max:
            ignored.append(IgnoredConstraint(field="property_ranges", value=f"{r.property} [{r.min}, {r.max}]",
                                             reason="min is greater than max"))
        else:
            kept_ranges.append(r)
    g.property_ranges = kept_ranges
    g.application_domains = list(dict.fromkeys(g.application_domains))
    g.crystal_systems = list(dict.fromkeys(g.crystal_systems))
    return g, ignored


def merge_constraints(primary: QueryConstraints, fallback: QueryConstraints) -> tuple[QueryConstraints, list[str]]:
    """Fill fields the primary (LLM) parse left empty from the fallback (rules).

    Free models often return only part of the schema (e.g. dropping the
    domain and numeric range from "Manganese cathode materials with energy
    above hull below 0.02 eV/atom"). The rule parser is conservative - it
    misses nuance but rarely invents constraints - so it only ever fills
    gaps; it never overrides a value the LLM did return.
    Returns (merged, names of fields taken from the fallback).
    """
    merged = primary.model_copy(deep=True)
    filled: list[str] = []
    for name in ("application_domains", "property_classes", "include_elements", "exclude_elements",
                 "crystal_systems", "space_groups", "formula_patterns", "oxidation_states", "anchor_materials"):
        if not getattr(merged, name) and getattr(fallback, name):
            setattr(merged, name, list(getattr(fallback, name)))
            filled.append(name)
    have = {r.property for r in merged.property_ranges}
    extra = [r for r in fallback.property_ranges if r.property not in have]
    if extra:
        merged.property_ranges += extra
        filled.append("property_ranges")
    flags = {b.flag for b in merged.boolean_flags}
    extra_flags = [b for b in fallback.boolean_flags if b.flag not in flags]
    if extra_flags:
        merged.boolean_flags += extra_flags
        filled.append("boolean_flags")
    if merged.sort_by is None and fallback.sort_by:
        merged.sort_by, merged.sort_descending = fallback.sort_by, fallback.sort_descending
        filled.append("sort_by")
    if merged.top_k == QueryConstraints().top_k and fallback.top_k != merged.top_k:
        merged.top_k = fallback.top_k
        filled.append("top_k")
    return merged, filled


def parse_question(question: str, graph: Graph, use_llm: bool = True) -> ParsedQuery:
    """Parse + ground. Falls back to the rule parser if the LLM is unavailable."""
    from matgraph.llm import LLMUnavailable, llm_configured

    parser, model = "rules", None
    if use_llm and llm_configured():
        try:
            constraints, model = parse_with_llm(question)
            constraints, filled = merge_constraints(constraints, parse_with_rules(question))
            parser = "llm+rules" if filled else "llm"
        except LLMUnavailable as error:
            logger.warning("LLM parse unavailable, using rule-based parser: %s", error)
            constraints = parse_with_rules(question)
    else:
        constraints = parse_with_rules(question)

    # The rule parser puts the whole question in semantic_text; so does an
    # empty LLM semantic_text, because semantic seeding needs *some* text.
    if not constraints.semantic_text.strip():
        constraints.semantic_text = question

    grounded, ignored = ground_constraints(constraints, graph)
    return ParsedQuery(
        question=question, constraints=grounded, ignored_constraints=ignored,
        parser=parser, parser_model=model,
    )


def constraints_summary(c: QueryConstraints) -> str:
    """Compact JSON of only the non-default fields (for prompts and logs)."""
    data = c.model_dump(exclude_defaults=True)
    data.pop("semantic_text", None)
    return json.dumps(data, sort_keys=True)
