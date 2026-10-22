"""Ask the materials knowledge graph a question.

Usage:
    uv run python scripts/ask.py "stable cobalt-free Li cathodes with band gap above 2 eV"
    uv run python scripts/ask.py "..." --no-llm      # rule parser + templated explanation
    uv run python scripts/ask.py "..." --json        # full Answer as JSON
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


def print_answer(answer) -> None:
    p = answer.parsed
    print(f"\nQuestion: {answer.question}")
    print(f"Parser: {p.parser}{f' ({p.parser_model})' if p.parser_model else ''}")
    print(f"Constraints: {p.constraints.model_dump(exclude_defaults=True, exclude={'semantic_text'})}")
    for ig in p.ignored_constraints:
        print(f"  ignored {ig.field}={ig.value!r}: {ig.reason}")
    r = answer.retrieval
    print(f"\nRetrieval: mode={r.mode}, {r.total_matching} matching of {r.pool_size} reached"
          + (f", excluded for missing data: {r.excluded_missing_data}" if r.excluded_missing_data else ""))
    for i, c in enumerate(r.candidates, 1):
        comps = ", ".join(f"{k}={v:.2f}" for k, v in c.score_components.items())
        flag = f"  [missing: {', '.join(c.missing_properties)}]" if c.missing_properties else ""
        print(f"  {i:2d}. {c.formula:16s} {c.material_id:14s} score={c.score:.3f} ({comps}){flag}")

    e = answer.explanation
    checks = {(c.section, c.text): c for c in answer.grounding.claims}

    def show(claim, section: str) -> None:
        check = checks.get((section, claim.text))
        mark = "✓" if check and check.grounded else "✗"
        cites = ",".join(claim.evidence_ids)
        print(f"   {mark} {claim.text} [{cites}]")
        if check and not check.grounded:
            print(f"       ! {'; '.join(check.problems)}")

    print(f"\nExplanation ({answer.generator}{f', {answer.generator_model}' if answer.generator_model else ''}):")
    for claim in e.summary:
        show(claim, "summary")
    for rec in e.recommendations:
        print(f"  {rec.formula} ({rec.material_id})")
        for claim in rec.claims:
            show(claim, f"recommendation:{rec.material_id}")
    if e.caveats:
        print("  Caveats:")
        for claim in e.caveats:
            show(claim, "caveat")
    g = answer.grounding
    print(f"\nTraceability: {g.grounded_claims}/{g.total_claims} claims grounded ({g.traceability:.0%})")
    for notice in answer.notices:
        print(f"Note: {notice}")
    print(f"Timings (ms): {answer.timings_ms}")


async def main(args) -> None:
    from matgraph.answer import answer_question
    from matgraph.graph_store import load_knowledge_base

    kb = await load_knowledge_base()
    answer = await answer_question(
        kb, args.question, use_llm=not args.no_llm, top_k=args.k, use_semantic=not args.no_semantic
    )
    if args.json:
        print(answer.model_dump_json(indent=2))
    else:
        print_answer(answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--no-llm", action="store_true", help="rule-based parser + templated explanation")
    parser.add_argument("--no-semantic", action="store_true", help="skip embedding similarity")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--k", type=int, default=None, help="number of candidates")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    for noisy in ("cognee", "litellm", "LiteLLM"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    asyncio.run(main(args))
