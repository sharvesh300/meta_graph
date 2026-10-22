"""Interpretation of raw CGCNN outputs, legacy-id mapping and eval metrics (no torch needed)."""

from types import SimpleNamespace

import pytest

from matgraph.cgcnn.interpret import METAL_THRESHOLD, interpret
from matgraph.cgcnn.registry import MODELS, legacy_to_new_id
from matgraph.eval.cgcnn import classification_stats, regression_stats, roc_auc


def raw_outputs(**values):
    defaults = {"formation-energy-per-atom": -2.0, "band-gap": 1.5, "semi-metal-classification": 0.05,
                "efermi": 3.0, "bulk-moduli": 2.0, "shear-moduli": 1.5, "poisson-ratio": 0.3}
    defaults.update({k.replace("_", "-"): v for k, v in values.items()})
    return {spec.name: SimpleNamespace(value=defaults[spec.name], flags=(), model_id=spec.model_id)
            for spec in MODELS}


@pytest.mark.parametrize("legacy, new", [("mp-149", "mp-aaaaaaft"), ("mp-754118", "mp-aaabqxoo"),
                                         ("mp-3799", "mp-aaaaafqd"), ("mp-0", "mp-aaaaaaaa")])
def test_legacy_ids_map_to_new_format(legacy, new):
    assert legacy_to_new_id(legacy) == new


def test_semiconductor_keeps_regressor_gap():
    v = interpret(raw_outputs())
    assert v["predicted_is_metal"].value is False
    assert v["predicted_band_gap"].value == pytest.approx(1.5)
    assert v["predicted_band_gap"].model_ids == ("cgcnn-txie93-f42ab23/semi-metal-classification",
                                                 "cgcnn-txie93-f42ab23/band-gap")


def test_metal_probability_at_threshold_gates_gap_to_zero():
    v = interpret(raw_outputs(semi_metal_classification=METAL_THRESHOLD, band_gap=2.2))
    assert v["predicted_is_metal"].value is True
    assert v["predicted_band_gap"].value == 0.0
    assert v["predicted_band_gap"].detail["regressor_output"] == pytest.approx(2.2)


def test_negative_regressor_gap_is_clipped():
    assert interpret(raw_outputs(band_gap=-0.3))["predicted_band_gap"].value == 0.0


def test_moduli_are_log10_gpa():
    v = interpret(raw_outputs(bulk_moduli=2.0, shear_moduli=1.5))
    assert v["predicted_bulk_modulus_vrh"].value == pytest.approx(100.0)
    assert v["predicted_shear_modulus_vrh"].value == pytest.approx(10 ** 1.5)
    assert v["predicted_bulk_modulus_vrh"].units == "GPa"


def test_expected_errors_come_from_committed_eval():
    v = interpret(raw_outputs())
    assert set(v) == {spec.attribute for spec in MODELS}
    for value in v.values():
        assert value.expected_error is not None and value.expected_error > 0, value.attribute
    assert v["predicted_bulk_modulus_vrh"].expected_error_units == "log10(GPa)"


def test_regression_and_classification_metrics():
    s = regression_stats([(1.0, 0.0), (3.0, 4.0)])
    assert s["mae"] == 1.0 and s["bias"] == 0.0 and s["n"] == 2
    assert roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert roc_auc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)
    c = classification_stats([0.9, 0.2, 0.6, 0.1], [1, 1, 0, 0], threshold=0.5)
    assert c["accuracy"] == 0.5 and c["precision_metal"] == 0.5 and c["recall_metal"] == 0.5
