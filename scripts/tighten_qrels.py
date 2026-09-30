"""
Narrows each qrel relevant moment from its whole ground-truth turn to the span of its
`matched_text`.

Span timing comes only from the reference side: the ground-truth turn text is force-aligned to
the audio inside the turn's own boundaries (wav2vec2 via WhisperX), and the words covering
`matched_text` give the span. Pipeline outputs are used solely as an independent QA check
(where does the pipeline transcript place the same words?) and never set labels.

Idempotent: the original turn bounds are kept in `turn_start_seconds` / `turn_end_seconds`.

Usage: uv run python -m scripts.tighten_qrels [--qrels path ...] [--dry-run]   (default: dev and test splits)
"""

import argparse
import difflib
import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import whisperx
from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH

from evals.qrels import SPLIT_PATHS
from src.pipeline.common import GROUND_TRUTH_DIR, OUTPUT_DIR, write_json

AUDIO_DIR = "dataset/audio"
SPAN_SOURCE = f"wav2vec2_forced_alignment_of_reference_turn ({DEFAULT_ALIGN_MODELS_TORCH['en']})"
QA_WARN_SECONDS = 3.0
PUNCTUATION = "\"'“”‘’.,!?;:()[]…-—"


@lru_cache(maxsize=1)
def _align_model():
    return whisperx.load_align_model(language_code="en", device="cpu")


@lru_cache(maxsize=None)
def _audio(file_id: str):
    return whisperx.load_audio(os.path.join(AUDIO_DIR, file_id))


@lru_cache(maxsize=None)
def align_reference_turn(file_id: str, text: str, start: float, end: float) -> Tuple[Tuple[str, Optional[float], Optional[float]], ...]:
    model, metadata = _align_model()
    result = whisperx.align([{"start": start, "end": end, "text": text}], model, metadata, _audio(file_id), "cpu")
    words = [w for seg in result["segments"] for w in seg["words"]]
    return tuple((w["word"], w.get("start"), w.get("end")) for w in words)


def word_char_spans(text: str, words: Tuple[str, ...]) -> List[Tuple[int, int]]:
    """Locates each aligned word in the original text, in order (robust to the aligner splitting a token)."""
    spans, cursor = [], 0
    for word in words:
        idx = text.find(word, cursor)
        if idx < 0:
            # The aligner can attach the same punctuation mark to both neighbours ('weakness."' and '"But').
            word = word.strip(PUNCTUATION)
            idx = text.find(word, cursor) if word else -1
        if idx < 0:
            raise RuntimeError(f"aligned word {word!r} not found in reference text after offset {cursor}")
        spans.append((idx, idx + len(word)))
        cursor = idx + len(word)
    return spans


def span_for_matched_text(file_id: str, turn_text: str, turn_start: float, turn_end: float, matched_text: str) -> Tuple[float, float]:
    char_start = turn_text.index(matched_text)
    char_end = char_start + len(matched_text)
    aligned = align_reference_turn(file_id, turn_text, turn_start, turn_end)
    spans = word_char_spans(turn_text, tuple(w for w, _, _ in aligned))

    covered = [(s_t, e_t) for (s, e), (_, s_t, e_t) in zip(spans, aligned) if s < char_end and e > char_start]
    starts = [s for s, _ in covered if s is not None]
    ends = [e for _, e in covered if e is not None]
    if not starts or not ends:
        raise RuntimeError(f"{file_id}: no aligned words inside matched_text '{matched_text[:40]}...'")
    return round(max(min(starts), turn_start), 2), round(min(max(ends), turn_end), 2)


def _norm_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def pipeline_span(file_id: str, matched_text: str, window: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    """QA only: locate matched_text in the pipeline transcript (within the turn window) by fuzzy token alignment."""
    path = os.path.join(OUTPUT_DIR, file_id.replace(".wav", "_canonical.json"))
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        words = [
            w for t in json.load(f)["turns"] for w in t["words"]
            if window[0] - 5.0 <= w["start_seconds"] <= window[1] + 5.0
        ]
    hyp: List[Tuple[str, Dict[str, Any]]] = [(tok, w) for w in words for tok in _norm_tokens(w["word"])]
    target = _norm_tokens(matched_text)
    matcher = difflib.SequenceMatcher(None, [h[0] for h in hyp], target, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size]
    if not blocks or sum(b.size for b in blocks) < 0.6 * len(target):
        return None
    first, last = blocks[0], blocks[-1]
    return hyp[first.a][1]["start_seconds"], hyp[last.a + last.size - 1][1]["end_seconds"]


def tighten(qrels_path: str, dry_run: bool = False) -> None:
    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels = json.load(f)

    transcripts: Dict[str, Dict[int, Dict[str, Any]]] = {}
    rows = []
    for q in qrels["queries"]:
        moments = [m for rm in q["relevant_moments"] for m in (rm, *rm.get("alternatives", ()))]
        for m in moments:
            fid = m["file_id"]
            if fid not in transcripts:
                with open(os.path.join(GROUND_TRUTH_DIR, fid.replace(".wav", ".json")), "r", encoding="utf-8") as f:
                    transcripts[fid] = {t["turn_id"]: t for t in json.load(f)["turns"]}
            turn = transcripts[fid][m["turn_id"]]
            turn_start = m.get("turn_start_seconds", turn["start_time"])
            turn_end = m.get("turn_end_seconds", turn["end_time"])

            start, end = span_for_matched_text(fid, turn["text"], float(turn_start), float(turn_end), m["matched_text"])
            m["turn_start_seconds"] = turn_start
            m["turn_end_seconds"] = turn_end
            m["start_seconds"] = start
            m["end_seconds"] = end
            m["span_source"] = SPAN_SOURCE

            qa = pipeline_span(fid, m["matched_text"], (float(turn_start), float(turn_end)))
            delta = None if qa is None else round(max(abs(qa[0] - start), abs(qa[1] - end)), 2)
            rows.append((q["query_id"], fid[:8], turn_end - turn_start, start, end, qa, delta))

    print(f"{'query':<6} {'file':<8} {'turn_s':>7} {'span':>17} {'pipeline_qa':>17} {'max_delta':>9}")
    for qid, fid, turn_dur, s, e, qa, delta in rows:
        qa_str = f"{qa[0]:.1f}-{qa[1]:.1f}" if qa else "n/a"
        flag = "  <-- check" if delta is None or delta > QA_WARN_SECONDS else ""
        print(f"{qid:<6} {fid:<8} {turn_dur:>7.1f} {f'{s:.1f}-{e:.1f}':>17} {qa_str:>17} {str(delta):>9}{flag}")

    if not dry_run:
        qrels["span_source"] = SPAN_SOURCE
        write_json(qrels_path, qrels)
        print(f"\nUpdated {qrels_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--qrels", action="append", help="Qrels file (repeatable). Defaults to every split.")
    args = parser.parse_args()
    for path in args.qrels or SPLIT_PATHS.values():
        tighten(path, args.dry_run)
