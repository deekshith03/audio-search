"""
Search chunks: sentence-bounded windows of about 30 s inside one speaker turn, with a
1-sentence overlap (the "A-30s" configuration chosen on the dev set; README §3.2).

Every chunk stays inside one turn, so it has exactly one speaker. Chunks are what retrieval
scores; results are then pinpointed to the best 1-3 sentences inside them (localize.py).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.search.sentences import Sentence

CHUNKER = "A-30s"
WINDOW_SECONDS = 30.0
MAX_WINDOW_STRETCH = 4 / 3

SETTINGS = {"chunker": CHUNKER, "window_seconds": WINDOW_SECONDS, "max_window_stretch": MAX_WINDOW_STRETCH}


@dataclass(frozen=True)
class Chunk:
    turn_id: int
    speaker_label: str
    start_s: float
    end_s: float
    text: str
    sentence_indexes: Tuple[int, ...]


def sentence_windows(sentences: List[Sentence], window_seconds: float = WINDOW_SECONDS) -> List[List[int]]:
    """Positions (within one turn) of each window's sentences.

    Windows grow while they fit in `window_seconds` (a single longer sentence is its own window).
    A window starts with the previous window's last sentence (1-sentence overlap) and must add at
    least one new sentence; a lone over-long sentence is not repeated as overlap. A final window
    whose new content is under a third of the target is folded into the one before it, as long as
    the result stays within MAX_WINDOW_STRETCH of the target.
    """
    n = len(sentences)
    windows: List[List[int]] = []
    start, overlap = 0, False
    while start < n:
        end = start + 1 if overlap else start
        while end + 1 < n and sentences[end + 1].end_s - sentences[start].start_s <= window_seconds:
            end += 1
        windows.append(list(range(start, end + 1)))
        if end == n - 1:
            break
        overlap = end > start
        start = end if overlap else end + 1
    if len(windows) >= 2:
        first_new = windows[-1][1] if windows[-1][0] == windows[-2][-1] else windows[-1][0]
        short_tail = sentences[-1].end_s - sentences[first_new].start_s < window_seconds / 3
        merged_fits = sentences[-1].end_s - sentences[windows[-2][0]].start_s <= window_seconds * MAX_WINDOW_STRETCH
        if short_tail and merged_fits:
            windows[-2] = list(range(windows[-2][0], n))
            windows.pop()
    return windows


def build_chunks(canonical: Dict[str, Any], sentences: List[Sentence]) -> List[Chunk]:
    """Chunks for every turn; `sentence_indexes` point into `sentences`."""
    by_turn: Dict[int, List[Tuple[int, Sentence]]] = {}
    for i, s in enumerate(sentences):
        by_turn.setdefault(s.turn_id, []).append((i, s))
    chunks: List[Chunk] = []
    for turn in canonical["turns"]:
        words = turn["words"]
        turn_sentences = by_turn.get(turn["turn_id"], [])
        if not words or not turn_sentences:
            continue
        in_turn = [s for _, s in turn_sentences]
        for window in sentence_windows(in_turn):
            first, last = in_turn[window[0]], in_turn[window[-1]]
            chunks.append(Chunk(
                turn_id=turn["turn_id"],
                speaker_label=turn["speaker_label"],
                start_s=words[first.word_start]["start_seconds"],
                end_s=words[last.word_end - 1]["end_seconds"],
                text=" ".join(w["word"] for w in words[first.word_start:last.word_end]),
                sentence_indexes=tuple(turn_sentences[p][0] for p in window),
            ))
    return chunks
