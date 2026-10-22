"""Ranking metrics: Recall@k, MRR, nDCG@k, and hard-constraint violation rate."""

from __future__ import annotations

import math


def recall_at_k(ranked: list[str], hits: set[str], k: int = 10) -> float:
    """|hits in top k| / min(k, |hits|) - capped so a query with 143 relevant
    materials can still score 1.0 when all 10 returned are relevant."""
    if not hits:
        return float("nan")
    return len(set(ranked[:k]) & hits) / min(k, len(hits))


def mrr(ranked: list[str], hits: set[str]) -> float:
    if not hits:
        return float("nan")
    for i, mid in enumerate(ranked, 1):
        if mid in hits:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], grades: dict[str, float], k: int = 10) -> float:
    if not grades:
        return float("nan")
    dcg = sum(grades.get(mid, 0.0) / math.log2(i + 1) for i, mid in enumerate(ranked[:k], 1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg else float("nan")


def violation_rate(ranked: list[str], satisfied: dict[str, float], k: int = 10) -> float:
    """Share of the top k that break at least one hard constraint."""
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for mid in top if satisfied.get(mid, 0.0) < 1.0) / len(top)


def mean(values: list[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else float("nan")
