"""Evaluate the pre-trained CGCNN models against the graph's DFT values (Phase 3 M1).

Usage:
    uv run python scripts/evaluate_cgcnn.py

Needs the converted weights and dataset id lists (scripts/fetch_cgcnn_weights.py)
and the cached structures in data/raw/. Writes artifacts/eval/cgcnn_graph_eval.{json,md};
the JSON's expected_errors are what matgraph.cgcnn.interpret attaches to predictions.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from matgraph.eval.cgcnn import JSON_PATH, MD_PATH, run

    report = run()
    for attribute, err in report["expected_errors"].items():
        print(f"  {attribute:38s} expected error {err['value']:.4f} {err['units']:12s} ({err['basis']}, n={err['n']})")
    print(f"wrote {MD_PATH}\n      {JSON_PATH}")
