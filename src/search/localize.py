"""
Pinpoints each retrieved chunk to its best 1-3 contiguous sentences, then drops overlapping hits.

The scorer only accepts short spans (targets have a median of 10.6 s; a correct 75 s chunk still
fails), so a chunk is never returned whole. Span selection starts at the best-scoring sentence of
the chunk and grows toward the better neighbour (neighbours may come from just outside the chunk,
within the same turn) while that neighbour scores at least `extend_ratio` × the best, or while the
span is still shorter than `min_seconds`, never beyond `max_sentences` or `max_seconds`.

For short keyword queries the span is tightened further to the matched words (± padding), since
a whole sentence can be far longer than a one-word target. A span overlapping a higher-ranked one
in the same file is dropped.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Set, Tuple

FUZZY_WORD_RATIO = 0.8
MAX_MATCH_WORDS = 3
STOPWORDS = {"a", "an", "and", "the", "of", "to", "in", "on", "for", "is", "are", "was", "it", "at", "by", "or"}
NON_ALNUM = re.compile(r"[^0-9a-z]+")


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


def seed_index(window: Sequence[ScoredSentence], eligible: Set[int]) -> int:
    """Position of the best-scoring sentence among the chunk's own (`eligible`) sentences."""
    seeds = [i for i, s in enumerate(window) if s.id in eligible]
    if not seeds:
        raise ValueError("chunk has no sentences in the window")
    return max(seeds, key=lambda i: (window[i].score, -i))


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
    best = seed_index(window, eligible)
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


def normalize_token(text: str) -> str:
    return NON_ALNUM.sub("", text.lower())


def query_terms(query: str) -> List[str]:
    """Content words of the query, plus the whole query joined (so "pgMustard" can match "PG Mustard")."""
    words = [normalize_token(w) for w in query.split()]
    terms = [w for w in words if w and w not in STOPWORDS]
    joined = "".join(words)
    if joined and joined not in terms:
        terms.append(joined)
    return terms


def match_score(candidate: str, terms: Sequence[str]) -> float:
    """1.0 for an exact term match, the fuzzy ratio when >= FUZZY_WORD_RATIO, else 0."""
    if len(candidate) < 2:
        return 0.0
    best = 0.0
    for term in terms:
        if candidate == term:
            return 1.0
        if min(len(candidate), len(term)) >= 4:
            ratio = SequenceMatcher(None, candidate, term).ratio()
            if ratio >= FUZZY_WORD_RATIO:
                best = max(best, ratio)
    return best


def keyword_matches(words: Sequence[Tuple[str, float, float]], query: str) -> Set[int]:
    """Positions of the words matching a query term, exactly or fuzzily.

    Candidate matches are runs of 1-3 words (so "PG Mustard" can match "pgMustard"); the best
    ones are taken greedily without overlap, exact before fuzzy and shorter before longer, so a
    neighbouring word is never pulled in just because "of PG Mustard" is also a fuzzy match.
    """
    terms = query_terms(query)
    if not terms or not words:
        return set()
    tokens = [normalize_token(w[0]) for w in words]
    candidates = []
    for i in range(len(tokens)):
        for n in range(1, MAX_MATCH_WORDS + 1):
            if i + n > len(tokens):
                break
            score = match_score("".join(tokens[i:i + n]), terms)
            if score:
                candidates.append((-score, n, i))
    taken: Set[int] = set()
    for _, n, i in sorted(candidates):
        span = set(range(i, i + n))
        if not span & taken:
            taken |= span
    return taken


def tighten_to_keywords(
    words: Sequence[Tuple[str, float, float]], query: str, padding: float
) -> Optional[Tuple[int, int]]:
    """Inclusive word positions covering every keyword_matches() hit plus `padding` seconds on
    each side, snapped to word boundaries; None if nothing matches."""
    taken = keyword_matches(words, query)
    if not taken:
        return None
    window_start = words[min(taken)][1] - padding
    window_end = words[max(taken)][2] + padding
    lo = next(i for i, w in enumerate(words) if w[2] > window_start)
    hi = max(i for i, w in enumerate(words) if w[1] < window_end)
    return lo, hi


def overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def dedupe(spans: Sequence[Tuple[Tuple, float, float]]) -> List[int]:
    """Indexes of spans to keep, in order: a span is dropped if it overlaps a kept one in the same file."""
    kept: List[int] = []
    for i, (file_key, start, end) in enumerate(spans):
        if not any(spans[j][0] == file_key and overlaps(start, end, spans[j][1], spans[j][2]) for j in kept):
            kept.append(i)
    return kept
