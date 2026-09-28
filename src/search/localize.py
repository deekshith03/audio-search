"""
Pinpoints each retrieved chunk to its best 1-3 contiguous sentences, then drops overlapping hits.

The scorer only accepts short spans (targets have a median of 10.6 s; a correct 75 s chunk still
fails), so a chunk is never returned whole. Span selection starts at the best-scoring sentence of
the chunk and grows toward the better neighbour (neighbours may come from just outside the chunk,
within the same turn) while that neighbour scores at least `extend_ratio` × the best, or while the
span is still shorter than `min_seconds`, never beyond `max_sentences` or `max_seconds`.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class ScoredSentence:
    id: int
    start_s: float
    end_s: float
    text: str
    score: float


def normalize(values: Sequence[Optional[float]]) -> List[float]:
    present = [v for v in values if v is not None]
    if not present:
        return [0.0] * len(values)
    lo, hi = min(present), max(present)
    if hi == lo:
        return [0.0 if v is None else 1.0 if hi > 0 else 0.0 for v in values]
    return [0.0 if v is None else (v - lo) / (hi - lo) for v in values]


def blend(signals: Dict[str, Sequence[Optional[float]]], weights: Dict[str, float]) -> List[float]:
    """Weighted sum of per-signal min-max normalized scores; each signal is one value per sentence."""
    n = len(next(iter(signals.values()))) if signals else 0
    total = [0.0] * n
    weight_sum = sum(weights.get(name, 0.0) for name in signals) or 1.0
    for name, values in signals.items():
        w = weights.get(name, 0.0) / weight_sum
        for i, v in enumerate(normalize(values)):
            total[i] += w * v
    return total


def select_span(
    window: Sequence[ScoredSentence],
    eligible: Set[int],
    max_sentences: int = 3,
    max_seconds: float = 20.0,
    min_seconds: float = 4.0,
    extend_ratio: float = 0.8,
) -> Tuple[int, int]:
    """Inclusive (lo, hi) positions in `window` (one turn's sentences, in order).

    The seed must be one of the chunk's own sentences (`eligible` ids); growth may use neighbours.
    """
    seeds = [i for i, s in enumerate(window) if s.id in eligible]
    if not seeds:
        raise ValueError("chunk has no sentences in the window")
    best = max(seeds, key=lambda i: (window[i].score, -i))
    lo = hi = best
    threshold = extend_ratio * window[best].score
    while hi - lo + 1 < max_sentences:
        options = [i for i in (lo - 1, hi + 1) if 0 <= i < len(window)]
        if not options:
            break
        nxt = max(options, key=lambda i: (window[i].score, -i))
        new_lo, new_hi = min(lo, nxt), max(hi, nxt)
        if window[new_hi].end_s - window[new_lo].start_s > max_seconds:
            break
        too_short = window[hi].end_s - window[lo].start_s < min_seconds
        if window[nxt].score < threshold and not too_short:
            break
        lo, hi = new_lo, new_hi
    return lo, hi


def overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def dedupe(spans: Sequence[Tuple[str, float, float]]) -> List[int]:
    """Indexes of spans to keep, in order: a span is dropped if it overlaps a kept one in the same file."""
    kept: List[int] = []
    for i, (file_key, start, end) in enumerate(spans):
        if not any(spans[j][0] == file_key and overlaps(start, end, spans[j][1], spans[j][2]) for j in kept):
            kept.append(i)
    return kept
