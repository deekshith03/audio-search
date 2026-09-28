"""
Splits canonical speaker turns into sentences: the smallest unit search results are pinpointed to.

A sentence never crosses a turn, so it always has exactly one speaker. Splitting runs in three
passes inside each turn:

1. punctuation: cut after words ending in . ? ! (known abbreviations excepted);
2. length: a piece over MAX_SENTENCE_WORDS words or MAX_SENTENCE_SECONDS is cut at its longest
   pause (>= MIN_PAUSE_SECONDS), repeatedly; with no usable pause it is cut every
   MAX_SENTENCE_WORDS words. ASR often drops punctuation entirely (87/110 segments in audio_02),
   so this pass does most of the work there;
3. fragments shorter than MIN_SENTENCE_WORDS words are merged into a neighbour in the same turn.

Timestamps come from the first and last word, so a sentence spans exactly the speech it contains.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

MAX_SENTENCE_WORDS = 30
MAX_SENTENCE_SECONDS = 15.0
MIN_PAUSE_SECONDS = 0.4
MIN_SENTENCE_WORDS = 4

SENTENCE_END_CHARS = (".", "?", "!")
TRAILING_CLOSERS = "\"')]”’"
ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.", "vs.", "etc.", "e.g.", "i.e.",
    "u.s.", "u.k.", "a.m.", "p.m.", "no.", "approx.", "inc.", "ltd.", "co.",
}

SETTINGS = {
    "max_sentence_words": MAX_SENTENCE_WORDS,
    "max_sentence_seconds": MAX_SENTENCE_SECONDS,
    "min_pause_seconds": MIN_PAUSE_SECONDS,
    "min_sentence_words": MIN_SENTENCE_WORDS,
}


@dataclass(frozen=True)
class Sentence:
    turn_id: int
    speaker_label: str
    word_start: int
    word_end: int
    start_s: float
    end_s: float
    text: str


def ends_sentence(word: str) -> bool:
    stripped = word.rstrip(TRAILING_CLOSERS)
    return stripped.endswith(SENTENCE_END_CHARS) and stripped.lower() not in ABBREVIATIONS


def split_on_punctuation(words: List[Dict[str, Any]]) -> List[range]:
    pieces, start = [], 0
    for i, w in enumerate(words):
        if ends_sentence(w["word"]):
            pieces.append(range(start, i + 1))
            start = i + 1
    if start < len(words):
        pieces.append(range(start, len(words)))
    return pieces


def is_too_long(words: List[Dict[str, Any]], piece: range) -> bool:
    duration = words[piece[-1]]["end_seconds"] - words[piece[0]]["start_seconds"]
    return len(piece) > MAX_SENTENCE_WORDS or duration > MAX_SENTENCE_SECONDS


def best_pause_split(words: List[Dict[str, Any]], piece: range) -> int:
    """Index of the first word after the longest usable pause, or -1 if there is none.

    Prefers cuts that leave MIN_SENTENCE_WORDS on both sides, so pause splits do not create
    fragments that pass 3 would immediately merge back.
    """
    best, best_gap = -1, MIN_PAUSE_SECONDS
    for i in range(piece.start + 1, piece.stop):
        if min(i - piece.start, piece.stop - i) < MIN_SENTENCE_WORDS:
            continue
        gap = words[i]["start_seconds"] - words[i - 1]["end_seconds"]
        if gap >= best_gap:
            best, best_gap = i, gap
    return best


def split_long(words: List[Dict[str, Any]], piece: range) -> List[range]:
    if not is_too_long(words, piece):
        return [piece]
    cut = best_pause_split(words, piece)
    if cut == -1:
        if len(piece) <= MAX_SENTENCE_WORDS:
            return [piece]
        cut = piece.start + MAX_SENTENCE_WORDS
    return split_long(words, range(piece.start, cut)) + split_long(words, range(cut, piece.stop))


def merge_fragments(pieces: List[range]) -> List[range]:
    merged: List[range] = []
    for piece in pieces:
        if merged and (len(piece) < MIN_SENTENCE_WORDS or len(merged[-1]) < MIN_SENTENCE_WORDS):
            merged[-1] = range(merged[-1].start, piece.stop)
        else:
            merged.append(piece)
    return merged


def split_turn(turn: Dict[str, Any]) -> List[Sentence]:
    words = turn["words"]
    if not words:
        return []
    pieces = [p for piece in split_on_punctuation(words) for p in split_long(words, piece)]
    return [
        Sentence(
            turn_id=turn["turn_id"],
            speaker_label=turn["speaker_label"],
            word_start=p.start,
            word_end=p.stop,
            start_s=words[p.start]["start_seconds"],
            end_s=words[p.stop - 1]["end_seconds"],
            text=" ".join(w["word"] for w in words[p.start:p.stop]),
        )
        for p in merge_fragments(pieces)
    ]


def split_transcript(canonical: Dict[str, Any]) -> List[Sentence]:
    return [s for turn in canonical["turns"] for s in split_turn(turn)]
