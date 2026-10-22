"""Build the Phase 1 materials knowledge graph and verify it.

Usage:
    uv run python scripts/build_graph.py [--no-reset] [--skip-visualize] [--skip-user-materials]
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

ARTIFACTS_DIR = ROOT / "artifacts"
GRAPH_HTML = ARTIFACTS_DIR / "graph.html"


async def main(reset: bool, skip_visualize: bool, replay_user: bool) -> None:
    from matgraph.pipeline import DATASET_NAME, run as run_pipeline
    from matgraph.verify import verify
    from cognee import visualize_graph

    await run_pipeline(reset=reset)

    if replay_user:
        from matgraph.graph_store import load_knowledge_base
        from matgraph.ingest import replay_user_materials

        replayed = await replay_user_materials(await load_knowledge_base())
        if replayed:
            logging.info("replayed %d persisted user materials: %s", len(replayed), ", ".join(replayed))

    if not skip_visualize:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        await visualize_graph(str(GRAPH_HTML), dataset=DATASET_NAME)
        logging.info("graph visualization written to %s", GRAPH_HTML)

    await verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-reset", dest="reset", action="store_false", help="don't prune cognee state first"
    )
    parser.add_argument("--skip-visualize", action="store_true")
    parser.add_argument("--skip-user-materials", dest="replay_user", action="store_false",
                        help="don't re-add materials persisted with scripts/add_material.py --persist")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    asyncio.run(main(reset=args.reset, skip_visualize=args.skip_visualize, replay_user=args.replay_user))
