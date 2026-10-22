from matgraph.query_parser import _normalize_species, ground_constraints, parse_with_rules
from matgraph.query_schema import QueryConstraints


def ranges(c):
    return {r.property: (r.min, r.max) for r in c.property_ranges}


def classes(c):
    return {(pc.kind, pc.name) for pc in c.property_classes}


def test_multi_constraint_cathode_question():
    c = parse_with_rules("stable, cobalt-free Li cathode with a band gap above 2 eV")
    assert c.application_domains == ["battery_cathode"]
    assert ranges(c) == {"band_gap": (2.0, None)}
    assert ("stability", "stable") in classes(c)
    assert c.include_elements == ["Li"]
    assert c.exclude_elements == ["Co"]


def test_between_range_and_structure_family():
    c = parse_with_rules("perovskites with band gap between 1.5 and 3 eV that are good photocatalysts")
    assert ranges(c) == {"band_gap": (1.5, 3.0)}
    assert c.formula_patterns == ["ABC3"]
    assert c.application_domains == ["photocatalyst"]


def test_anchor_and_element_suffix():
    c = parse_with_rules("materials similar to LiFePO4 but Mn-based")
    assert c.anchor_materials == ["LiFePO4"]
    assert c.include_elements == ["Mn"]


def test_ordering_words_vs_domain():
    c = parse_with_rules("antiferromagnetic iron oxides")
    assert ("magnetic_ordering", "AFM") in classes(c)
    assert "magnetic_material" not in c.application_domains
    assert set(c.include_elements) == {"Fe", "O"}


def test_dielectric_constant_is_a_property_not_the_domain():
    c = parse_with_rules("materials containing Ti4+ with dielectric constant above 20")
    assert ranges(c) == {"dielectric_total": (20.0, None)}
    assert c.application_domains == []
    assert c.oxidation_states == ["Ti4+"]


def test_li_ion_is_not_a_lithium_requirement():
    c = parse_with_rules("Li-ion cathodes containing Fe and Mn")
    assert "Li" not in c.include_elements
    assert set(c.include_elements) == {"Fe", "Mn"}


def test_superlatives_and_top_k():
    c = parse_with_rules("top 5 materials with the highest bulk modulus")
    assert c.top_k == 5
    assert c.sort_by == "bulk_modulus_vrh" and c.sort_descending


def test_species_normalization():
    assert _normalize_species("Ti+4") == "Ti4+"
    assert _normalize_species("li1+") == "Li+"
    assert _normalize_species("O2-") == "O2-"


def test_grounding_reports_unmappable_values(graph):
    raw = QueryConstraints(
        include_elements=["lithium", "Xx"],
        exclude_elements=["Li"],
        formula_patterns=["ABC3", "A9B9"],
        oxidation_states=["Ti+4"],
        anchor_materials=["LiFePO4", "Unobtainium"],
    )
    grounded, ignored = ground_constraints(raw, graph)
    fields = {(i.field, i.value) for i in ignored}
    assert grounded.include_elements == []  # Li is also excluded: exclusion wins
    assert ("include_elements", "Xx") in fields
    assert ("include_elements", "Li") in fields
    assert grounded.formula_patterns == ["ABC3"]
    assert ("formula_patterns", "A9B9") in fields
    assert grounded.oxidation_states == ["Ti4+"]
    assert grounded.anchor_materials == ["LiFePO4"]
    assert ("anchor_materials", "Unobtainium") in fields


def test_free_of_never_includes_the_element():
    c = parse_with_rules("Candidates for photocatalysis that are free of lead")
    assert c.exclude_elements == ["Pb"]
    assert "Pb" not in c.include_elements


def test_llm_omissions_filled_from_rules_without_overriding():
    from matgraph.query_parser import merge_constraints

    llm = QueryConstraints(include_elements=["Mn"])  # free model dropped domain + range
    rules = parse_with_rules("Manganese cathode materials with energy above hull below 0.02 eV/atom")
    merged, filled = merge_constraints(llm, rules)
    assert merged.application_domains == ["battery_cathode"]
    assert ranges(merged) == {"energy_above_hull": (None, 0.02)}
    assert merged.include_elements == ["Mn"]  # LLM value kept
    assert set(filled) == {"application_domains", "property_ranges"}
