"""
Combine ranked lists from several retrievers into one ranking.

    rrf:    score = Σ wᵢ / (k + rankᵢ)                 ranks only; robust default (k = 60)
    convex: score = Σ wᵢ · minmaxᵢ(scoreᵢ)             normalized scores; Bruch et al. found it beats
                                                        RRF when the weights are tuned
A chunk missing from a list contributes nothing for that list. Ties break by chunk id so the
order is deterministic.
"""

from typing import Dict, List, Sequence, Tuple

Ranked = Sequence[Tuple[int, float]]


def rrf(lists: Dict[str, Ranked], weights: Dict[str, float], k: int = 60) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = {}
    for name, ranked in lists.items():
        w = weights.get(name, 0.0)
        for rank, (item, _) in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def min_max(ranked: Ranked) -> Dict[int, float]:
    if not ranked:
        return {}
    values = [s for _, s in ranked]
    lo, hi = min(values), max(values)
    if hi == lo:
        return {item: 1.0 for item, _ in ranked}
    return {item: (s - lo) / (hi - lo) for item, s in ranked}


def convex(lists: Dict[str, Ranked], weights: Dict[str, float]) -> List[Tuple[int, float]]:
    active = {name: weights.get(name, 0.0) for name in lists}
    total = sum(active.values()) or 1.0
    scores: Dict[int, float] = {}
    for name, ranked in lists.items():
        w = active[name] / total
        for item, s in min_max(ranked).items():
            scores[item] = scores.get(item, 0.0) + w * s
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


FUSIONS = {"rrf", "convex"}


def fuse(method: str, lists: Dict[str, Ranked], weights: Dict[str, float], k: int = 60) -> List[Tuple[int, float]]:
    if method == "rrf":
        return rrf(lists, weights, k)
    if method == "convex":
        return convex(lists, weights)
    raise ValueError(f"unknown fusion method: {method}")
