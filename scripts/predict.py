"""Predict properties of crystal structures with the pre-trained CGCNN models.

Usage:
    uv run python scripts/predict.py my.cif [POSCAR ...] [--json] [--raw]

Prints the graph-ready values (band gap gated by the metal classifier, moduli
in GPa) with the expected error measured on unseen graph materials; --raw also
shows each model's direct output. Nothing is added to the graph (use
scripts/add_material.py --structure for that). Needs the [ml] extra and
scripts/fetch_cgcnn_weights.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _error_text(v) -> str:
    if v.expected_error is None:
        return ""
    if v.expected_error_units.startswith("log10"):
        return f"(×/÷ {10 ** v.expected_error:.2f})"
    if v.expected_error_units == "error rate":
        return f"({v.expected_error:.0%} misclassified)"
    return f"(±{v.expected_error:.3g})"


def main(paths: list[Path], as_json: bool, show_raw: bool) -> int:
    from matgraph.cgcnn.interpret import interpret
    from matgraph.cgcnn.predictor import CGCNNPredictor
    from matgraph.cgcnn.validate import StructureError, allowed_elements, check_structure, read_structure_file, struct_id
    from matgraph.enrich import load_config, load_elements_table

    config, elements_table = load_config(), load_elements_table()
    allowed, excluded = allowed_elements(elements_table), set(config["exclude_elements"])
    predictor = CGCNNPredictor()
    results, failed = [], 0
    for path in paths:
        try:
            structure = read_structure_file(path)
            check_structure(structure, allowed, excluded)
        except (OSError, StructureError) as error:
            print(f"{path}: error: {error}", file=sys.stderr)
            failed += 1
            continue
        raw = predictor.predict(structure)
        values = interpret(raw)
        results.append((path, structure, raw, values))

    if as_json:
        print(json.dumps([
            {"file": str(path), "formula": s.composition.reduced_formula, "struct_id": struct_id(s), "nsites": len(s),
             "predictions": {name: {"value": v.value, "units": v.units, "expected_error": v.expected_error,
                                    "expected_error_units": v.expected_error_units, "model_ids": list(v.model_ids),
                                    "detail": v.detail, "flags": list(v.flags)} for name, v in values.items()},
             **({"raw": {n: p.value for n, p in raw.items()}} if show_raw else {})}
            for path, s, raw, values in results
        ], indent=2))
        return 1 if failed else 0

    for path, s, raw, values in results:
        print(f"\n{path}: {s.composition.reduced_formula}, {len(s)} sites -> {struct_id(s)}")
        for name, v in values.items():
            shown = v.value if isinstance(v.value, bool) else f"{v.value:.4g}"
            extra = f"  P(metal) = {v.detail['p_metal']:.2f}" if "p_metal" in v.detail else ""
            print(f"  {name:38s} {shown!s:>10} {v.units:8s} {_error_text(v)}{extra}")
        if any(v.flags for v in values.values()):
            print(f"  flags: {', '.join(sorted({f for v in values.values() for f in v.flags}))}")
        if show_raw:
            for name, p in raw.items():
                print(f"  raw {name:34s} {p.value:10.4g} {p.units}")
    print("\nCGCNN predictions (txie-93 pre-trained models), not DFT values. Expected errors: "
          "artifacts/eval/cgcnn_graph_eval.md")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+", type=Path, help="CIF / POSCAR files")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--raw", action="store_true", help="also show each model's direct output")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", module="pymatgen")
    sys.exit(main(args.paths, args.json, args.raw))
