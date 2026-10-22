"""Turn raw CGCNN outputs into graph-ready predicted values.

Settled in M1 against the graph's DFT values (artifacts/eval/cgcnn_graph_eval.md):
- the moduli models predict log10(GPa) (MAE 0.066 vs 2.6 when read as ln);
- the band-gap regressor was trained on non-metals only and predicts ~1.6 eV
  for DFT metals, so the gap is gated by the metal classifier. The threshold
  was chosen on the materials the models had seen (P(metal) >= 0.3: MAE
  0.41 eV vs 0.48 at 0.5) and is reported on the unseen ones.

Each value carries the expected error measured on graph materials outside
the model's dataset, read from the committed evaluation JSON.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from matgraph.cgcnn.registry import ROOT

METAL_THRESHOLD = 0.3
MODULI_LOG_BASE = 10.0
EVAL_PATH = ROOT / "artifacts" / "eval" / "cgcnn_graph_eval.json"


@dataclass(frozen=True)
class PredictedValue:
    attribute: str  # Material attribute, e.g. "predicted_band_gap"
    value: float | bool
    units: str
    model_ids: tuple[str, ...]
    expected_error: float | None  # MAE on unseen graph materials (error rate for is_metal)
    expected_error_units: str
    detail: dict[str, Any] = field(default_factory=dict)
    flags: tuple[str, ...] = ()


@functools.cache
def expected_errors(path: Path = EVAL_PATH) -> dict[str, dict[str, Any]]:
    """attribute -> {"value", "units", "n", "basis"}; empty before M1's eval has run."""
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("expected_errors", {})


def interpret(raw: dict[str, Any]) -> dict[str, PredictedValue]:
    """raw: model name -> predictor.Prediction (all registry models present)."""
    errors = expected_errors()
    flags = tuple(sorted({f for p in raw.values() for f in p.flags}))

    def value(attribute, v, units, models, **detail):
        err = errors.get(attribute, {})
        return PredictedValue(
            attribute=attribute, value=v, units=units, model_ids=tuple(raw[m].model_id for m in models),
            expected_error=err.get("value"), expected_error_units=err.get("units", units), detail=detail,
            flags=flags,
        )

    p_metal = raw["semi-metal-classification"].value
    is_metal = p_metal >= METAL_THRESHOLD
    gap_raw = raw["band-gap"].value
    out = [
        value("predicted_formation_energy_per_atom", raw["formation-energy-per-atom"].value, "eV/atom",
              ["formation-energy-per-atom"]),
        value("predicted_is_metal", is_metal, "", ["semi-metal-classification"],
              p_metal=p_metal, threshold=METAL_THRESHOLD),
        value("predicted_band_gap", 0.0 if is_metal else max(gap_raw, 0.0), "eV",
              ["semi-metal-classification", "band-gap"], regressor_output=gap_raw, gated_by_metal=is_metal),
        value("predicted_efermi", raw["efermi"].value, "eV", ["efermi"]),
        value("predicted_bulk_modulus_vrh", MODULI_LOG_BASE ** raw["bulk-moduli"].value, "GPa", ["bulk-moduli"],
              log10_gpa=raw["bulk-moduli"].value),
        value("predicted_shear_modulus_vrh", MODULI_LOG_BASE ** raw["shear-moduli"].value, "GPa", ["shear-moduli"],
              log10_gpa=raw["shear-moduli"].value),
        value("predicted_homogeneous_poisson", raw["poisson-ratio"].value, "", ["poisson-ratio"]),
    ]
    return {v.attribute: v for v in out}
