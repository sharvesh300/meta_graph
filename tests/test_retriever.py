from matgraph.query_schema import BooleanConstraint, PropertyClassRef, PropertyRange, QueryConstraints
from matgraph.retriever import retrieve


def formulas(result):
    return [c.formula for c in result.candidates]


def test_hard_constraints_are_all_enforced(graph, retrieval_config):
    c = QueryConstraints(
        application_domains=["battery_cathode"],
        property_ranges=[PropertyRange(property="band_gap", min=3.0)],
        exclude_elements=["Co"],
    )
    result = retrieve(graph, c, config=retrieval_config)
    assert set(formulas(result)) == {"LiFePO4", "LiMnPO4"}
    for cand in result.candidates:
        labels = {ch.constraint for ch in cand.checks}
        assert "suitable for battery_cathode" in labels
        assert all(ch.satisfied for ch in cand.checks)


def test_every_node_constraint_carries_its_graph_path(graph, retrieval_config):
    c = QueryConstraints(crystal_systems=["Hexagonal"], include_elements=["N"])
    result = retrieve(graph, c, config=retrieval_config)
    assert formulas(result) == ["GaN"]
    cs_check = next(ch for ch in result.candidates[0].checks if ch.constraint.startswith("crystal system"))
    assert [h.relation for h in cs_check.path] == ["has_space_group", "crystal_system"]
    assert cs_check.path[-1].target == "CrystalSystem:Hexagonal"


def test_rule_text_travels_with_suitable_for_path(graph, retrieval_config):
    result = retrieve(graph, QueryConstraints(application_domains=["dielectric"]), config=retrieval_config)
    hop = result.candidates[0].checks[0].path[0]
    assert hop.relation == "suitable_for"
    assert "band_gap > 2 eV" in hop.properties["rule"]


def test_missing_values_excluded_and_counted(graph, retrieval_config):
    c = QueryConstraints(property_ranges=[PropertyRange(property="bulk_modulus_vrh", min=90)])
    result = retrieve(graph, c, config=retrieval_config)
    assert set(formulas(result)) == {"LiMnPO4", "GaN", "BaTiO3"}
    assert result.excluded_missing_data == {"bulk_modulus_vrh": 3}


def test_keep_flagged_policy_ranks_incomplete_last(graph, retrieval_config):
    retrieval_config["missing_value_policy"] = "keep_flagged"
    c = QueryConstraints(property_ranges=[PropertyRange(property="bulk_modulus_vrh", min=90)])
    result = retrieve(graph, c, config=retrieval_config)
    flagged = [cand for cand in result.candidates if cand.missing_properties]
    assert len(flagged) == 3
    assert result.candidates[-1].missing_properties == ["bulk_modulus_vrh"]


def test_anchor_query_excludes_anchor_and_prefers_similar(graph, retrieval_config):
    c = QueryConstraints(anchor_materials=["LiFePO4"])
    result = retrieve(graph, c, config=retrieval_config)
    assert "LiFePO4" not in formulas(result)
    assert formulas(result)[0] == "LiMnPO4"
    link = result.candidates[0].anchor_links[0]
    assert link.similar_to and link.similar_to[0].properties["weight"] == 0.95
    assert "FormulaPattern:ABCD4" in link.shared_hubs


def test_boolean_flags_and_classes(graph, retrieval_config):
    c = QueryConstraints(
        boolean_flags=[BooleanConstraint(flag="is_gap_direct", value=True)],
        property_classes=[PropertyClassRef(kind="stability", name="stable")],
    )
    assert formulas(retrieve(graph, c, config=retrieval_config)) == ["GaN"]


def test_sort_by_property(graph, retrieval_config):
    c = QueryConstraints(include_elements=["O"], sort_by="density", sort_descending=True, top_k=2)
    assert formulas(retrieve(graph, c, config=retrieval_config)) == ["BaTiO3", "Fe3O4"]


def test_unknown_node_constraint_yields_no_results(graph, retrieval_config):
    c = QueryConstraints(include_elements=["Zr"])
    result = retrieve(graph, c, config=retrieval_config)
    assert result.candidates == [] and result.total_matching == 0


def test_semantic_scores_order_ties(graph, retrieval_config):
    c = QueryConstraints(application_domains=["battery_cathode"])
    ids = {graph.node[n]["formula"]: n for n in graph.by_type["Material"]}
    semantic = {ids["LiCoO2"]: 0.9, ids["LiFePO4"]: 0.5, ids["LiMnPO4"]: 0.4}
    assert formulas(retrieve(graph, c, semantic, retrieval_config))[0] == "LiCoO2"
