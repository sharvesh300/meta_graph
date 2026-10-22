"""include_predicted in retrieval, evidence and grounding (Phase 3 M3).

The graph is the synthetic build from test_ingest_structure plus one preview
struct- node made with a fake CGCNN predictor (wurtzite MgO, predicted gap
2.0 eV, non-metal)."""

import asyncio

import pytest

import matgraph.ingest_structure as ist
from matgraph.evidence import PREDICTION_CAVEAT, build_evidence
from matgraph.generator import Claim, Explanation, Recommendation, explain_with_template
from matgraph.grounding import check_explanation
from matgraph.query_schema import BooleanConstraint, PropertyClassRef, PropertyRange, QueryConstraints
from matgraph.retriever import retrieve

from test_ingest_structure import kb, wurtzite  # noqa: F401  (kb is a fixture)


@pytest.fixture
def graph_with_struct(kb):  # noqa: F811
    report = asyncio.run(ist.ingest_structure(kb, wurtzite("Mg", "O")))
    return kb.graph, report.material_id


def formulas(result):
    return [(c.formula, c.material_id) for c in result.candidates]


def test_default_retrieval_never_sees_predictions(graph_with_struct):
    g, struct = graph_with_struct
    for c in [QueryConstraints(include_elements=["Mg"]),
              QueryConstraints(property_classes=[PropertyClassRef(kind="band_gap", name="semiconductor")]),
              QueryConstraints(property_ranges=[PropertyRange(property="band_gap", min=1.0)]),
              QueryConstraints()]:
        result = retrieve(g, c)
        assert struct not in {mid for _, mid in formulas(result)}, c
        assert not any(cand.uses_predictions for cand in result.candidates)


def test_opt_in_uses_predicted_edges_and_values_and_marks_them(graph_with_struct):
    g, struct = graph_with_struct
    by_class = retrieve(g, QueryConstraints(property_classes=[PropertyClassRef(kind="band_gap", name="semiconductor")]),
                        include_predicted=True)
    cand = next(c for c in by_class.candidates if c.material_id == struct)
    assert cand.uses_predictions and cand.checks[0].predicted
    assert cand.checks[0].path[0].properties["provenance"] == "predicted"

    by_range = retrieve(g, QueryConstraints(property_ranges=[PropertyRange(property="band_gap", min=1.5, max=2.5)]),
                        include_predicted=True)
    check = next(c for c in by_range.candidates if c.material_id == struct).checks[0]
    assert check.predicted and check.value == pytest.approx(2.0)
    # GaN (DFT 1.7 eV) is in range on its DFT value, not marked predicted.
    gan = next(c for c in by_range.candidates if c.formula == "GaN")
    assert not gan.uses_predictions and not gan.checks[0].predicted

    metal = retrieve(g, QueryConstraints(include_elements=["Mg"],
                                         boolean_flags=[BooleanConstraint(flag="is_metal", value=False)]),
                     include_predicted=True)
    struct_check = next(c for c in metal.candidates if c.material_id == struct).checks[-1]
    assert struct_check.predicted and struct_check.value is False


def test_evidence_labels_predictions_and_template_is_grounded(graph_with_struct):
    g, struct = graph_with_struct
    c = QueryConstraints(include_elements=["Mg"])
    result = retrieve(g, c, include_predicted=True)
    evidence = build_evidence(g, result, c, include_predicted=True)
    assert any(f.text == PREDICTION_CAVEAT for f in evidence.facts)
    predicted = [f for f in evidence.facts if f.material_id == struct and f.predicted]
    gap = next(f for f in predicted if "band_gap" in f.values)
    assert "predicted band_gap ≈ 2.0 eV" in gap.text and "not a DFT value" in gap.text
    assert gap.values["expected_error"] > 0
    assert any("[predicted" in f.text for f in predicted if f.kind == "edge")
    # Without the flag the same material's facts carry no predicted values.
    plain = build_evidence(g, retrieve(g, c), c)
    assert not any(f.predicted for f in plain.facts)

    explanation = explain_with_template(evidence)
    report = check_explanation(explanation, evidence, g, [x.material_id for x in result.candidates], "Mg materials")
    assert report.traceability == 1.0


def test_grounding_flags_unhedged_predicted_numbers(graph_with_struct):
    g, struct = graph_with_struct
    c = QueryConstraints(include_elements=["Mg"])
    result = retrieve(g, c, include_predicted=True)
    evidence = build_evidence(g, result, c, include_predicted=True)
    gap = next(f for f in evidence.facts if f.material_id == struct and f.predicted and "band_gap" in f.values)

    def check(text):
        explanation = Explanation(summary=[], caveats=[], recommendations=[
            Recommendation(material_id=struct, formula="MgO", claims=[Claim(text=text, evidence_ids=[gap.id])])])
        return check_explanation(explanation, evidence, g, [struct], "Mg materials").claims[0]

    bad = check("MgO has a band gap of 2.0 eV.")
    assert not bad.grounded and "without saying it is predicted" in bad.problems[0]
    assert check("MgO has a predicted band gap of 2.0 eV.").grounded
    assert check("CGCNN estimates a band gap ≈ 2.0 eV for MgO.").grounded


def test_struct_node_does_not_shift_default_ranking(kb):  # noqa: F811
    """Ranking statistics (anchor hub IDF, property spreads) ignore struct-
    nodes unless predictions are opted in, so default scores are unchanged."""
    queries = [QueryConstraints(anchor_materials=["GaN"]),
               QueryConstraints(property_ranges=[PropertyRange(property="density", min=3.0)])]
    before = [[(c.material_id, c.score) for c in retrieve(kb.graph, q).candidates] for q in queries]
    asyncio.run(ist.ingest_structure(kb, wurtzite("Mg", "O")))  # shares P6_3mc with GaN, has a density
    after = [[(c.material_id, c.score) for c in retrieve(kb.graph, q).candidates] for q in queries]
    assert after == before
