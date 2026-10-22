"""Run the Phase 2 evaluation and write artifacts/eval/results.{json,md}.

Usage:
    uv run python scripts/evaluate.py                    # retrieval configs only (no LLM calls)
    uv run python scripts/evaluate.py --llm-sample 12    # + parser accuracy, end-to-end, traceability
    uv run python scripts/evaluate.py --ingestion 50     # + leave-one-out ingestion check
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


async def main(args) -> None:
    from matgraph.eval.run import OUT_DIR, run_evaluation
    from matgraph.graph_store import load_knowledge_base

    kb = await load_knowledge_base()
    results = await run_evaluation(kb, llm_sample=args.llm_sample)
    if args.ingestion:
        from matgraph.eval.ingestion import run_ingestion_eval

        results["ingestion"] = await run_ingestion_eval(args.ingestion)
        from matgraph.eval.run import _json_default, render_markdown
        import json

        (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=_json_default))
        md = render_markdown(results)
        md += "\n## Ingestion (leave-one-out)\n\n" + "\n".join(
            f"- {k}: {v}" for k, v in results["ingestion"]["summary"].items()) + "\n"
        (OUT_DIR / "results.md").write_text(md)
    print((OUT_DIR / "results.md").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm-sample", type=int, default=0,
                        help="number of benchmark queries to run through the LLM (free-tier budget)")
    parser.add_argument("--ingestion", type=int, default=0, help="leave-one-out ingestion sample size")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    for noisy in ("cognee", "litellm", "LiteLLM"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    asyncio.run(main(args))
