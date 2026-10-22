"""Add a new material to the knowledge graph and show how it relates.

Usage:
    uv run python scripts/add_material.py LaAlO3                # preview (not saved)
    uv run python scripts/add_material.py mp-2920 --persist     # write to the graph
    uv run python scripts/add_material.py LaAlO3 --json

Accepts a Materials Project id or a formula. Crystal structures (CIF) are not
accepted: structure input goes to the CGCNN model in Phase 3.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")


def print_report(r) -> None:
    status = "already in graph" if r.already_in_graph else ("persisted" if r.persisted else "preview")
    print(f"\n{r.formula} ({r.material_id}) - source: {r.source}, {status}")
    print("\nProperties (value [provenance]):")
    for p in r.properties:
        if p.value is None:
            continue
        print(f"  {p.name:28s} {p.value!s:40.40s} [{p.provenance}]")
    unavailable = [p.name for p in r.properties if p.provenance == "unavailable"]
    if unavailable:
        print(f"  unavailable: {', '.join(unavailable)}")
    print("\nShared graph nodes (most selective first):")
    for h in r.shared_hubs[:12]:
        tag = "NEW NODE" if h.new_node else f"shared with {h.shared_with} materials"
        print(f"  -{h.relation}-> {h.node:40s} {tag}")
    if r.similar_materials:
        print("\nsimilar_to (10-d property cosine):")
        for n in r.similar_materials:
            print(f"  {n.formula:16s} {n.material_id:14s} {n.weight:.3f}")
    if r.compositional_neighbours:
        print("\nCompositional neighbours (element-fraction cosine):")
        for n in r.compositional_neighbours:
            print(f"  {n.formula:16s} {n.material_id:14s} {n.weight:.3f}")
    if r.rules:
        print("\nApplication-domain rules:")
        for rule in r.rules:
            print(f"  {rule.domain:22s} {rule.status:16s} {rule.reason}")
    for note in r.notes:
        print(f"Note: {note}")


async def main(args) -> None:
    from matgraph.graph_store import load_knowledge_base
    from matgraph.ingest import IngestError, ingest_material

    kb = await load_knowledge_base()
    try:
        report = await ingest_material(kb, args.material, persist=args.persist)
    except IngestError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
    print(report.model_dump_json(indent=2) if args.json else "", end="")
    if not args.json:
        print_report(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("material", help="Materials Project id (mp-...) or formula")
    parser.add_argument("--persist", action="store_true", help="write to the graph (default: preview only)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    for noisy in ("cognee", "litellm", "LiteLLM"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    asyncio.run(main(args))
