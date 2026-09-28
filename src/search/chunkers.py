"""
Chunk configurations compared on the dev set (docs/PHASE_3_PLAN.md §4). Every chunk stays inside
one speaker turn, so each has exactly one speaker.

    A-15s / A-30s / A-45s   sentence-bounded windows of about that length, 1-sentence overlap
    B                       one sentence per chunk
    C-512                   512 bge tokens cut at word boundaries, 64-token overlap (textbook default)
    D                       one chunk per turn

A and B carry `context_text`: the tail of the previous turn by the other speaker, used only as
embedding input so an answer like "Unfortunately, no" is embedded together with its question.
Keyword search indexes `text` alone, so keyword hits stay attributed to the right speaker.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.search.sentences import Sentence

CONTEXT_MAX_WORDS = 60
C_MAX_TOKENS = 510
C_OVERLAP_TOKENS = 64
C_TOKENIZER = "BAAI/bge-small-en-v1.5"

CHUNK_CONFIGS = ("A-15s", "A-30s", "A-45s", "B", "C-512", "D")
CONTEXT_CONFIGS = {"A-15s", "A-30s", "A-45s", "B"}
WINDOW_SECONDS = {"A-15s": 15.0, "A-30s": 30.0, "A-45s": 45.0}
MAX_WINDOW_STRETCH = 4 / 3

SETTINGS = {
    "configs": CHUNK_CONFIGS,
    "context_max_words": CONTEXT_MAX_WORDS,
    "max_window_stretch": MAX_WINDOW_STRETCH,
    "c_max_tokens": C_MAX_TOKENS,
    "c_overlap_tokens": C_OVERLAP_TOKENS,
    "c_tokenizer": C_TOKENIZER,
}

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class Chunk:
    chunker: str
    turn_id: int
    speaker_label: str
    start_s: float
    end_s: float
    text: str
    context_text: Optional[str]
    sentence_indexes: Tuple[int, ...]


def bge_token_counter() -> TokenCounter:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(C_TOKENIZER)
    return lambda word: len(tokenizer.tokenize(word))


def previous_other_speaker_context(turns: Sequence[Dict[str, Any]], turn_index: int) -> Optional[str]:
    speaker = turns[turn_index]["speaker_label"]
    for prev in reversed(turns[:turn_index]):
        if prev["speaker_label"] != speaker and prev["words"]:
            return " ".join(w["word"] for w in prev["words"][-CONTEXT_MAX_WORDS:])
    return None


def sentence_windows(sentences: List[Sentence], window_seconds: float) -> List[List[int]]:
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


def token_windows(words: List[Dict[str, Any]], count_tokens: TokenCounter) -> List[range]:
    counts = [count_tokens(w["word"]) for w in words]
    windows: List[range] = []
    start = 0
    while start < len(words):
        end, total = start, 0
        while end < len(words) and (total + counts[end] <= C_MAX_TOKENS or end == start):
            total += counts[end]
            end += 1
        windows.append(range(start, end))
        if end == len(words):
            break
        overlap_start, overlap = end, 0
        while overlap_start - 1 > start and overlap + counts[overlap_start - 1] <= C_OVERLAP_TOKENS:
            overlap_start -= 1
            overlap += counts[overlap_start]
        start = overlap_start
    return windows


def chunk_turn(
    config: str,
    turn: Dict[str, Any],
    context: Optional[str],
    turn_sentences: List[Tuple[int, Sentence]],
    count_tokens: Optional[TokenCounter],
) -> List[Chunk]:
    words = turn["words"]
    if not words:
        return []

    def make(word_range: range, sentence_indexes: Sequence[int], with_context: bool) -> Chunk:
        return Chunk(
            chunker=config,
            turn_id=turn["turn_id"],
            speaker_label=turn["speaker_label"],
            start_s=words[word_range.start]["start_seconds"],
            end_s=words[word_range.stop - 1]["end_seconds"],
            text=" ".join(w["word"] for w in words[word_range.start:word_range.stop]),
            context_text=context if with_context else None,
            sentence_indexes=tuple(sentence_indexes),
        )

    def overlapping_sentences(word_range: range) -> List[int]:
        return [i for i, s in turn_sentences if s.word_start < word_range.stop and s.word_end > word_range.start]

    if config == "D":
        return [make(range(0, len(words)), [i for i, _ in turn_sentences], False)]
    if config == "B":
        return [make(range(s.word_start, s.word_end), [i], True) for i, s in turn_sentences]
    if config in WINDOW_SECONDS:
        sentences = [s for _, s in turn_sentences]
        return [
            make(
                range(sentences[w[0]].word_start, sentences[w[-1]].word_end),
                [turn_sentences[p][0] for p in w],
                True,
            )
            for w in sentence_windows(sentences, WINDOW_SECONDS[config])
        ]
    if config == "C-512":
        if count_tokens is None:
            raise ValueError("C-512 needs a token counter")
        return [make(r, overlapping_sentences(r), False) for r in token_windows(words, count_tokens)]
    raise ValueError(f"unknown chunk config: {config}")


def build_chunks(
    canonical: Dict[str, Any],
    sentences: List[Sentence],
    configs: Sequence[str] = CHUNK_CONFIGS,
    count_tokens: Optional[TokenCounter] = None,
) -> List[Chunk]:
    """Chunks for every config; `sentence_indexes` point into `sentences`."""
    turns = canonical["turns"]
    by_turn: Dict[int, List[Tuple[int, Sentence]]] = {}
    for i, s in enumerate(sentences):
        by_turn.setdefault(s.turn_id, []).append((i, s))
    chunks: List[Chunk] = []
    for config in configs:
        for turn_index, turn in enumerate(turns):
            context = previous_other_speaker_context(turns, turn_index) if config in CONTEXT_CONFIGS else None
            chunks.extend(chunk_turn(config, turn, context, by_turn.get(turn["turn_id"], []), count_tokens))
    return chunks


def embedding_input(chunk_text: str, context_text: Optional[str], with_context: bool) -> str:
    if with_context and context_text:
        return f"{context_text}\n\n{chunk_text}"
    return chunk_text
