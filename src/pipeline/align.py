"""
Stage 1B: Word-Level Forced Alignment via Wav2Vec2 (WhisperX aligner, CPU).

Refines Whisper word timestamps to CTC frame boundaries (~20 ms). WhisperX aligns characters
outside the wav2vec2 vocabulary (digits, "$", "%") through a wildcard token, so those words are
still acoustically timed; they are counted as `wildcard_aligned_words`. Words that receive no
timing at all are spread evenly across the gap between their timed neighbours and tagged
`interpolated_fallback`. A segment WhisperX returns with no words at all (e.g. a 0.1 s "Right."
it could not align) keeps its text the same way, spread across the segment, instead of vanishing
from the transcript.
"""

import json
import os
import time
from functools import lru_cache
from importlib.metadata import version
from typing import Any, Dict, List, Tuple

import whisperx
from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH

from src.pipeline.common import (
    RAW_ASR_DIR,
    build_cache_key,
    file_base,
    is_cache_valid,
    parse_stage_args,
    resolve_audio_files,
    write_json,
)

LANGUAGE = "en"
DEVICE = "cpu"
ALIGN_MODEL = DEFAULT_ALIGN_MODELS_TORCH[LANGUAGE]
MIN_INTERPOLATED_WORD_SECONDS = 0.05


def align_config() -> Dict[str, Any]:
    return {
        "stage": "align",
        "aligner": "whisperx",
        "aligner_version": version("whisperx"),
        "model": ALIGN_MODEL,
        "language": LANGUAGE,
        "device": DEVICE,
        "min_interpolated_word_seconds": MIN_INTERPOLATED_WORD_SECONDS,
        "unaligned_segment_fallback": "segment_text",
    }


@lru_cache(maxsize=1)
def load_align_model() -> Tuple[Any, Dict[str, Any]]:
    return whisperx.load_align_model(language_code=LANGUAGE, device=DEVICE)


def is_timed(word: Dict[str, Any]) -> bool:
    return word.get("start") is not None and word.get("end") is not None


def interpolate_untimed_words(
    words: List[Dict[str, Any]],
    seg_start: float,
    seg_end: float,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Converts WhisperX word dicts into the canonical word schema. Each run of consecutive untimed
    words is spread evenly between the previous timed word's end and the next timed word's start
    (segment bounds at the edges). Returns (words, number_of_interpolated_words).
    """
    out: List[Dict[str, Any]] = []
    interpolated = 0
    i = 0
    while i < len(words):
        w = words[i]
        if is_timed(w):
            score = w.get("score")
            out.append({
                "word": w.get("word", "").strip(),
                "start_seconds": round(float(w["start"]), 3),
                "end_seconds": round(float(w["end"]), 3),
                "confidence": round(float(score), 3) if score is not None else None,
                "timing_source": "wav2vec2_aligned",
            })
            i += 1
            continue

        run_end = i
        while run_end < len(words) and not is_timed(words[run_end]):
            run_end += 1
        run = words[i:run_end]

        gap_start = out[-1]["end_seconds"] if out else seg_start
        gap_end = float(words[run_end]["start"]) if run_end < len(words) else seg_end
        gap_end = max(gap_end, gap_start + MIN_INTERPOLATED_WORD_SECONDS * len(run))
        step = (gap_end - gap_start) / len(run)

        for k, uw in enumerate(run):
            out.append({
                "word": uw.get("word", "").strip(),
                "start_seconds": round(gap_start + k * step, 3),
                "end_seconds": round(gap_start + (k + 1) * step, 3),
                "confidence": None,
                "timing_source": "interpolated_fallback",
            })
        interpolated += len(run)
        i = run_end

    return out, interpolated


def segment_words(seg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The segment's aligned words, or its text as untimed words when alignment returned none."""
    return seg.get("words") or [{"word": token} for token in seg.get("text", "").split()]


def is_wildcard_word(word_text: str, dictionary: Dict[str, int]) -> bool:
    letters = [c for c in word_text.lower() if not c.isspace()]
    return bool(letters) and not any(c in dictionary for c in letters)


def align_file(
    audio_path: str,
    raw_asr_path: str,
    output_dir: str = RAW_ASR_DIR,
    force: bool = False,
) -> str:
    base = file_base(audio_path)
    out_file = os.path.join(output_dir, f"{base}_aligned.json")
    config = align_config()
    cache_key = build_cache_key(config, [audio_path, raw_asr_path])

    if not force and is_cache_valid(out_file, cache_key):
        print(f"[{base}] Using cached aligned ASR: {out_file}")
        return out_file

    with open(raw_asr_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_segments = raw_data.get("segments", [])
    if not raw_segments:
        raise ValueError(f"[{base}] No ASR segments in {raw_asr_path}; cannot align.")

    align_model, align_metadata = load_align_model()
    audio = whisperx.load_audio(audio_path)

    print(f"[{base}] Aligning {len(raw_segments)} segments with {ALIGN_MODEL} ({DEVICE})...")
    start_t = time.time()
    aligned_result = whisperx.align(
        [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in raw_segments],
        align_model,
        align_metadata,
        audio,
        DEVICE,
        return_char_alignments=False,
    )
    elapsed = time.time() - start_t

    dictionary = align_metadata["dictionary"]
    total_words = 0
    fallback_words = 0
    wildcard_words = 0
    processed_segments = []

    for seg in aligned_result.get("segments", []):
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        words, n_interp = interpolate_untimed_words(segment_words(seg), seg_start, seg_end)
        total_words += len(words)
        fallback_words += n_interp
        wildcard_words += sum(1 for w in words if w["timing_source"] == "wav2vec2_aligned" and is_wildcard_word(w["word"], dictionary))
        processed_segments.append({
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "text": seg.get("text", "").strip(),
            "words": words,
        })

    fallback_rate = (fallback_words / total_words) if total_words else 0.0
    print(f"[{base}] Aligned in {elapsed:.1f}s | words={total_words} interpolated={fallback_words} ({fallback_rate:.2%}) wildcard={wildcard_words}")

    write_json(out_file, {
        "file_id": os.path.basename(audio_path),
        "asr_model": raw_data.get("model"),
        "alignment_model": ALIGN_MODEL,
        "config": config,
        "cache_key": cache_key,
        "runtime_seconds": round(elapsed, 2),
        "telemetry": {
            "total_words": total_words,
            "fallback_aligned_words": fallback_words,
            "alignment_fallback_rate": round(fallback_rate, 4),
            "wildcard_aligned_words": wildcard_words,
        },
        "segments": processed_segments,
    })
    return out_file


def run_batch_alignment(audio_files: List[str], asr_dir: str = RAW_ASR_DIR, force: bool = False) -> List[str]:
    print(f"Starting alignment on {len(audio_files)} file(s) ({DEVICE})...")
    out_files = []
    for audio_path in audio_files:
        raw_asr_path = os.path.join(asr_dir, f"{file_base(audio_path)}_raw.json")
        if not os.path.exists(raw_asr_path):
            raise FileNotFoundError(f"Raw ASR missing: {raw_asr_path}. Run src.pipeline.asr first.")
        out_files.append(align_file(audio_path, raw_asr_path, output_dir=asr_dir, force=force))
    return out_files


if __name__ == "__main__":
    args = parse_stage_args("Stage 1B: wav2vec2 forced alignment")
    run_batch_alignment(resolve_audio_files(args), asr_dir=args.workspace.raw_asr_dir, force=args.force)
