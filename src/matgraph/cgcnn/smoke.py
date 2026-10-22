"""Fixed-structure smoke test for the loaded CGCNN models (PRD: the API checks
that the model loads and produces the expected output on a fixed structure).

smoke_test.cif is COD 1000041 (NaCl) from txie-93/cgcnn's sample-regression
data; EXPECTED are the reference implementation's outputs on it with the
original checkpoints (tests/fixtures/cgcnn/reference_predictions.json, made by
scripts/make_cgcnn_reference_fixture.py).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

SMOKE_CIF = Path(__file__).resolve().parent / "smoke_test.cif"
EXPECTED = {
    "formation-energy-per-atom": -2.1132540702819824,
    "band-gap": 4.774056434631348,
    "semi-metal-classification": 4.423649443197064e-06,
    "efermi": -0.6838362216949463,
    "bulk-moduli": 1.3808047771453857,
    "shear-moduli": 0.9817742109298706,
    "poisson-ratio": 0.3124755322933197,
}
TOLERANCE = 1e-4


def smoke_test(predictor: Any) -> dict[str, Any]:
    from pymatgen.core import Structure

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pymatgen notes rounded CIF coordinates
        structure = Structure.from_file(SMOKE_CIF)
    got = {name: p.value for name, p in predictor.predict(structure).items()}
    diff = max(abs(got[name] - expected) for name, expected in EXPECTED.items())
    return {"ok": diff <= TOLERANCE, "max_abs_diff": diff, "structure": "COD 1000041 (NaCl)"}
