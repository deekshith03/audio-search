"""
Convex-combination fusion of ranked lists from several retrievers:

    score = Σ wᵢ · minmaxᵢ(scoreᵢ)      (weights normalized to sum to 1)

Each list is min-max normalized per query, so BM25's unbounded scores and cosine similarity are
comparable. It beat weighted reciprocal rank fusion on the dev set (docs/PHASE_3_PLAN.md §6), in
line with Bruch et al. A chunk missing from a list contributes nothing for that list. Ties break
by chunk id so the order is deterministic.
"""

from typing import Dict, List, Sequence, Tuple

Ranked = Sequence[Tuple[int, float]]


def min_max(ranked: Ranked) -> Dict[int, float]:
    if not ranked:
        return {}
    values = [s for _, s in ranked]
    lo, hi = min(values), max(values)
    if hi == lo:
        return {item: 1.0 for item, _ in ranked}
    return {item: (s - lo) / (hi - lo) for item, s in ranked}


def contributions(lists: Dict[str, Ranked], weights: Dict[str, float]) -> Dict[int, Dict[str, float]]:
    """Per item, what each list adds to its fused score (w · minmax); convex() sums these."""
    active = {name: weights.get(name, 0.0) for name in lists}
    total = sum(active.values()) or 1.0
    parts: Dict[int, Dict[str, float]] = {}
    for name, ranked in lists.items():
        w = active[name] / total
        for item, s in min_max(ranked).items():
            parts.setdefault(item, {})[name] = w * s
    return parts


def convex(lists: Dict[str, Ranked], weights: Dict[str, float]) -> List[Tuple[int, float]]:
    scores = {item: sum(p.values()) for item, p in contributions(lists, weights).items()}
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
