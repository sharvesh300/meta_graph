from matgraph.evidence import build_evidence
from matgraph.generator import Claim, Explanation, Recommendation, explain_with_template
from matgraph.grounding import check_explanation
from matgraph.query_schema import PropertyRange, QueryConstraints
from matgraph.retriever import retrieve


def setup(graph, retrieval_config, constraints=None):
    c = constraints or QueryConstraints(
        application_domains=["battery_cathode"], property_ranges=[PropertyRange(property="band_gap", min=3.0)]
    )
    result = retrieve(graph, c, config=retrieval_config)
    evidence = build_evidence(graph, result, c)
    return c, result, evidence


def fact_id(evidence, formula, text_fragment):
    return next(f.id for f in evidence.facts if f.formula == formula and text_fragment in f.text)


def check(graph, evidence, result, explanation, question="cathodes with band gap above 3 eV"):
    return check_explanation(explanation, evidence, graph, [c.material_id for c in result.candidates],
                             question, [3.0])


def test_template_explanation_is_fully_grounded(graph, retrieval_config):
    _, result, evidence = setup(graph, retrieval_config)
    report = check(graph, evidence, result, explain_with_template(evidence))
    assert report.traceability == 1.0 and report.total_claims > 0


def test_missing_properties_are_explicit_evidence(graph, retrieval_config):
    _, _, evidence = setup(graph, retrieval_config)
    missing = [f for f in evidence.facts if f.kind == "missing" and f.formula == "LiFePO4"]
    assert missing and "bulk_modulus_vrh" in missing[0].values["missing"]


def test_correct_citation_with_rounding_passes(graph, retrieval_config):
    _, result, evidence = setup(graph, retrieval_config)
    gap = fact_id(evidence, "LiMnPO4", "band_gap")
    explanation = Explanation(summary=[], caveats=[], recommendations=[Recommendation(
        material_id="mp-limnpo4", formula="LiMnPO4",
        claims=[Claim(text="LiMnPO4 has a band gap of 3.9 eV, above the requested 3 eV.", evidence_ids=[gap])],
    )])
    assert check(graph, evidence, result, explanation).traceability == 1.0


def test_invented_number_is_flagged(graph, retrieval_config):
    _, result, evidence = setup(graph, retrieval_config)
    gap = fact_id(evidence, "LiMnPO4", "band_gap")
    explanation = Explanation(summary=[], caveats=[], recommendations=[Recommendation(
        material_id="mp-limnpo4", formula="LiMnPO4",
        claims=[Claim(text="LiMnPO4 has a bulk modulus of 250 GPa.", evidence_ids=[gap])],
    )])
    report = check(graph, evidence, result, explanation)
    assert report.traceability == 0.0
    assert "number 250" in report.claims[0].problems[0]


def test_uncited_unknown_ids_and_unretrieved_material_are_flagged(graph, retrieval_config):
    _, result, evidence = setup(graph, retrieval_config)
    explanation = Explanation(
        summary=[Claim(text="Two cathodes qualify.", evidence_ids=[])],
        caveats=[Claim(text="Band gaps are DFT values.", evidence_ids=["E999"])],
        recommendations=[Recommendation(material_id="mp-gan", formula="GaN",
                                        claims=[Claim(text="GaN is great.", evidence_ids=["E1"])])],
    )
    report = check(graph, evidence, result, explanation)
    problems = [p for c in report.claims for p in c.problems]
    assert any("no evidence cited" in p for p in problems)
    assert any("unknown evidence ids" in p for p in problems)
    assert any("was not retrieved" in p for p in problems)
    assert report.unretrieved_recommendations == ["mp-gan"]
    assert report.traceability == 0.0


def test_mentioning_material_outside_evidence_is_flagged(graph, retrieval_config):
    _, result, evidence = setup(graph, retrieval_config)
    gap = fact_id(evidence, "LiFePO4", "band_gap")
    explanation = Explanation(summary=[], caveats=[], recommendations=[Recommendation(
        material_id="mp-lifepo4", formula="LiFePO4",
        claims=[Claim(text="LiFePO4 outperforms BaTiO3.", evidence_ids=[gap])],
    )])
    report = check(graph, evidence, result, explanation)
    assert "mentions BaTiO3" in report.claims[0].problems[0]


def test_identifier_digits_are_not_numbers(graph, retrieval_config):
    _, result, evidence = setup(graph, retrieval_config)
    edge = fact_id(evidence, "LiFePO4", "suitable_for")
    explanation = Explanation(summary=[], caveats=[], recommendations=[Recommendation(
        material_id="mp-lifepo4", formula="LiFePO4",
        claims=[Claim(text="LiFePO4 (Fe2+, P5+, space group P6_3mc-free) qualifies as a cathode.",
                      evidence_ids=[edge])],
    )])
    assert check(graph, evidence, result, explanation).traceability == 1.0
