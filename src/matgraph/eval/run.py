"""Evaluation harness (plan §7): retrieval comparison, parser accuracy,
end-to-end retrieval, and explanation traceability.

Retrieval configurations run from the **gold** constraints, so they need no
LLM and isolate the retrieval mechanism. The LLM parts run on a sample of
queries (OpenRouter free tier has daily request caps); every LLM response is
cached, so re-running is free and reproduces exactly.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import yaml

from matgraph.eval import baselines
from matgraph.eval.metrics import mean, mrr, ndcg_at_k, recall_at_k, violation_rate
from matgraph.eval.oracle import is_hit, load_oracle_materials, relevance, satisfied_fraction
from matgraph.graph_store import KnowledgeBase
from matgraph.query_schema import QueryConstraints
from matgraph.retriever import load_retrieval_config

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_PATH = ROOT / "config" / "benchmark_queries.yaml"
OUT_DIR = ROOT / "artifacts" / "eval"

CONFIGS = ["B1_vector_rag", "B2_graphrag_fixed_hop", "proposed", "ablation_no_semantic", "ablation_no_similar_to"]


def load_benchmark() -> list[dict[str, Any]]:
    return yaml.safe_load(BENCHMARK_PATH.read_text())["queries"]


def _score(ranked: list[str], grades: dict[str, float], satisfied: dict[str, float], anchored: bool, k: int) -> dict:
    hits = {m for m, g in grades.items() if is_hit(g, anchored)}
    return {
        "recall@10": recall_at_k(ranked, hits, k),
        "mrr": mrr(ranked, hits),
        "ndcg@10": ndcg_at_k(ranked, grades, k),
        "violation_rate": violation_rate(ranked, satisfied, k),
        "returned": len(ranked[:k]),
        "relevant": len(hits),
    }


def _field_values(c: QueryConstraints) -> dict[str, Any]:
    """Normalized, comparable form of each constraint field (defaults dropped)."""
    out: dict[str, Any] = {}
    if c.application_domains:
        out["application_domains"] = frozenset(c.application_domains)
    if c.property_ranges:
        out["property_ranges"] = frozenset((r.property, r.min, r.max) for r in c.property_ranges)
    if c.boolean_flags:
        out["boolean_flags"] = frozenset((b.flag, b.value) for b in c.boolean_flags)
    if c.property_classes:
        out["property_classes"] = frozenset((p.kind, p.name) for p in c.property_classes)
    for name in ("include_elements", "exclude_elements", "crystal_systems", "space_groups",
                 "formula_patterns", "oxidation_states", "anchor_materials"):
        value = getattr(c, name)
        if value:
            out[name] = frozenset(value)
    if c.sort_by:
        out["sort_by"] = (c.sort_by, c.sort_descending)
    return out


HELDOUT_PATH = ROOT / "config" / "parser_heldout.yaml"


def rule_parser_accuracy() -> dict[str, Any]:
    """Rule-based parser vs gold constraints. The benchmark set was used to
    fix the parser; the held-out paraphrase set was not (see its header)."""
    from matgraph.query_parser import parse_with_rules

    out = {}
    for name, path in (("benchmark", BENCHMARK_PATH), ("heldout", HELDOUT_PATH)):
        correct = total = exact = 0
        queries = yaml.safe_load(path.read_text())["queries"]
        for q in queries:
            c, t, wrong = parser_field_accuracy(QueryConstraints(**q["gold"]), parse_with_rules(q["question"]))
            correct, total, exact = correct + c, total + t, exact + (not wrong)
        out[name] = {"field_accuracy": round(correct / total, 4), "exact": f"{exact}/{len(queries)}"}
    return out


def parser_field_accuracy(gold: QueryConstraints, parsed: QueryConstraints) -> tuple[int, int, list[str]]:
    g, p = _field_values(gold), _field_values(parsed)
    fields = set(g) | set(p)
    wrong = sorted(f for f in fields if g.get(f) != p.get(f))
    return len(fields) - len(wrong), len(fields), wrong


async def run_evaluation(kb: KnowledgeBase, llm_sample: int = 0, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    oracle = load_oracle_materials()
    queries = load_benchmark()
    cfg = load_retrieval_config()
    g = kb.graph
    per_query: list[dict[str, Any]] = []

    for q in queries:
        gold = QueryConstraints(**q["gold"])
        k = gold.top_k
        anchored = bool(gold.anchor_materials)
        grades = relevance(oracle, gold)
        satisfied = {mid: satisfied_fraction(m, gold) for mid, m in oracle.items()}
        semantic = await kb.semantic_scores(q["question"], tuple(cfg["semantic_collections"]))

        t = time.perf_counter()
        runs = {
            "B1_vector_rag": baselines.vector_rag(g, semantic, k),
            "B2_graphrag_fixed_hop": baselines.graphrag_fixed_hop(g, gold, semantic, k),
            "proposed": baselines.proposed(g, gold, semantic, cfg, k),
            "ablation_no_semantic": baselines.proposed(g, gold, {}, cfg, k),
            "ablation_no_similar_to": baselines.proposed_no_similarity(g, gold, semantic, cfg, k),
        }
        elapsed = (time.perf_counter() - t) * 1000
        per_query.append({
            "id": q["id"], "tier": q["tier"], "question": q["question"], "relevant": len(
                [m for m, gr in grades.items() if is_hit(gr, anchored)]),
            "metrics": {name: _score(ranked, grades, satisfied, anchored, k) for name, ranked in runs.items()},
            "ranked": runs, "retrieval_ms_all_configs": round(elapsed, 1),
        })

    summary = _aggregate(per_query, CONFIGS)
    results: dict[str, Any] = {"summary": summary, "queries": per_query,
                               "rule_parser": rule_parser_accuracy()}

    if llm_sample:
        results["llm"] = await _run_llm_part(kb, queries[:: max(1, len(queries) // llm_sample)][:llm_sample],
                                             oracle, cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=_json_default))
    (out_dir / "results.md").write_text(render_markdown(results))
    return results


def _json_default(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return str(value)


def _aggregate(per_query: list[dict[str, Any]], configs: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    tiers = sorted({q["tier"] for q in per_query})
    for name in configs:
        entry: dict[str, Any] = {}
        for scope in ["all"] + tiers:
            rows = [q["metrics"][name] for q in per_query if scope == "all" or q["tier"] == scope]
            entry[str(scope)] = {m: round(mean([r[m] for r in rows]), 4)
                                 for m in ("recall@10", "mrr", "ndcg@10", "violation_rate")}
        summary[name] = entry
    return summary


async def _run_llm_part(kb: KnowledgeBase, sample: list[dict], oracle, cfg) -> dict[str, Any]:
    from matgraph.answer import answer_question

    rows = []
    for q in sample:
        gold = QueryConstraints(**q["gold"])
        anchored = bool(gold.anchor_materials)
        answer = await answer_question(kb, q["question"], use_llm=True, top_k=gold.top_k)
        correct, total, wrong = parser_field_accuracy(gold, answer.parsed.constraints)
        grades = relevance(oracle, gold)
        satisfied = {mid: satisfied_fraction(m, gold) for mid, m in oracle.items()}
        ranked = [c.material_id for c in answer.retrieval.candidates]
        rows.append({
            "id": q["id"], "tier": q["tier"],
            "parser": answer.parsed.parser, "parser_model": answer.parsed.parser_model,
            "parser_fields_correct": correct, "parser_fields_total": total, "parser_wrong_fields": wrong,
            "e2e_metrics": _score(ranked, grades, satisfied, anchored, gold.top_k),
            "generator": answer.generator, "generator_model": answer.generator_model,
            "traceability": answer.grounding.traceability,
            "claims": answer.grounding.total_claims,
            "ungrounded": [c.model_dump() for c in answer.grounding.claims if not c.grounded],
        })
    llm_parsed = [r for r in rows if r["parser"] in ("llm", "llm+rules")]
    llm_explained = [r for r in rows if r["generator"] == "llm"]
    return {
        "sample_size": len(rows),
        "parsed_by_llm": len(llm_parsed),
        "parser_field_accuracy": round(
            sum(r["parser_fields_correct"] for r in llm_parsed) / max(1, sum(r["parser_fields_total"] for r in llm_parsed)), 4
        ) if llm_parsed else None,
        "e2e": {m: round(mean([r["e2e_metrics"][m] for r in rows]), 4)
                for m in ("recall@10", "mrr", "ndcg@10", "violation_rate")},
        "explained_by_llm": len(llm_explained),
        "traceability_llm": round(
            sum(r["traceability"] * r["claims"] for r in llm_explained) / max(1, sum(r["claims"] for r in llm_explained)), 4
        ) if llm_explained else None,
        "rows": rows,
    }


def render_markdown(results: dict[str, Any]) -> str:
    s = results["summary"]
    lines = ["# Phase 2 retrieval evaluation", "",
             "Generated by `scripts/evaluate.py`. Relevance from the independent oracle over raw MP JSON; "
             "retrieval configs run on gold constraints (no LLM).", "",
             "## Overall (all tiers)", "",
             "| Config | Recall@10 | MRR | nDCG@10 | Constraint violations |", "|---|---|---|---|---|"]
    for name, entry in s.items():
        a = entry["all"]
        lines.append(f"| {name} | {a['recall@10']:.3f} | {a['mrr']:.3f} | {a['ndcg@10']:.3f} | {a['violation_rate']:.1%} |")
    tiers = [k for k in next(iter(s.values())) if k != "all"]
    names = {"1": "single constraint", "2": "multi-constraint", "3": "multi-hop", "4": "similarity-anchored",
             "5": "sparse data"}
    lines += ["", "## nDCG@10 by tier", "", "| Config | " + " | ".join(f"T{t} {names.get(t, '')}" for t in tiers) + " |",
              "|---|" + "---|" * len(tiers)]
    for name, entry in s.items():
        lines.append(f"| {name} | " + " | ".join(f"{entry[t]['ndcg@10']:.3f}" for t in tiers) + " |")
    if "rule_parser" in results:
        rp = results["rule_parser"]
        lines += ["", "## Rule-based parser", "",
                  f"- Benchmark questions (parser was fixed against these): field accuracy "
                  f"{rp['benchmark']['field_accuracy']:.1%}, exact {rp['benchmark']['exact']}",
                  f"- Held-out paraphrases (not tuned on): field accuracy "
                  f"{rp['heldout']['field_accuracy']:.1%}, exact {rp['heldout']['exact']}"]
    if "llm" in results:
        l = results["llm"]
        lines += ["", "## LLM components (sampled)", "",
                  f"- Sample: {l['sample_size']} queries; parsed by LLM: {l['parsed_by_llm']}; "
                  f"explained by LLM: {l['explained_by_llm']}",
                  f"- Parser field-level accuracy: {l['parser_field_accuracy']}",
                  f"- End-to-end (LLM-parsed constraints): {l['e2e']}",
                  f"- Explanation traceability (grounded claims / all claims, LLM-written only): {l['traceability_llm']}"]
    return "\n".join(lines) + "\n"
