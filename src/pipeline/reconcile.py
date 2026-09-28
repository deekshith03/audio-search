"""
Stage 3: Deterministic Word-to-Speaker Turn Reconciliation.

Rules:
1. Word assignment: a word goes to the speaker whose intervals cover strictly >50% of its
   duration; otherwise (low overlap or diarization gap) to the temporally nearest interval.
2. Smoothing: a single word whose speaker differs from identical neighbours on both sides, with
   <0.4 s gaps, is folded into the surrounding speaker (unless it is a backchannel word).
   Decisions read the original assignments, so corrections never cascade.
3. Turn splitting: speaker change -> "speaker_change"; pause > 1.5 s -> "pause";
   final turn -> "end_of_audio".
4. Short turns: < 0.8 s and matching the backchannel lexicon -> is_short_turn = True.
5. Speakers: only anonymous diarization labels (SPEAKER_00 / SPEAKER_01) are emitted. Human
   names come from the separate speaker-labeling step and are joined at read time.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import soundfile as sf

from src.pipeline.common import (
    DIARIZATION_DIR,
    OUTPUT_DIR,
    RAW_ASR_DIR,
    file_base,
    parse_stage_args,
    resolve_audio_files,
    write_json,
)

PIPELINE_VERSION = "2.1.0"
PAUSE_THRESHOLD_SECONDS = 1.5
SMOOTHING_MAX_GAP_SECONDS = 0.4
SHORT_TURN_MAX_SECONDS = 0.8
BACKCHANNEL_WORDS = {"yeah", "right", "okay", "sure", "yep", "mm-hmm", "uh-huh", "yes", "i see", "exactly"}


def _strip(text: str) -> str:
    return text.lower().strip(".,!?;:\"'")


def assign_word_to_speaker(word: Dict[str, Any], intervals: List[Dict[str, Any]]) -> str:
    """
    Assigns word to speaker with strictly >50% overlap.
    If overlap <= 50% or word is in a diarization gap, falls back to temporally nearest interval.
    """
    w_start = word["start_seconds"]
    w_end = word["end_seconds"]
    w_dur = max(w_end - w_start, 0.001)

    spk_overlaps: Dict[str, float] = {}
    for iv in intervals:
        overlap = max(0.0, min(w_end, iv["end"]) - max(w_start, iv["start"]))
        if overlap > 0:
            spk_overlaps[iv["speaker"]] = spk_overlaps.get(iv["speaker"], 0.0) + overlap

    if spk_overlaps:
        best_spk, max_overlap = max(spk_overlaps.items(), key=lambda x: x[1])
        if (max_overlap / w_dur) > 0.5:
            return best_spk

    nearest_spk = intervals[0]["speaker"]
    min_dist = float("inf")
    for iv in intervals:
        if w_end < iv["start"]:
            dist = iv["start"] - w_end
        elif w_start > iv["end"]:
            dist = w_start - iv["end"]
        else:
            dist = 0.0
        if dist < min_dist:
            min_dist = dist
            nearest_spk = iv["speaker"]
    return nearest_spk


def smooth_isolated_words(assignments: List[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
    smoothed = list(assignments)
    for i in range(1, len(assignments) - 1):
        prev_spk, w_prev = assignments[i - 1]
        curr_spk, w_curr = assignments[i]
        next_spk, w_next = assignments[i + 1]
        if prev_spk != next_spk or curr_spk == prev_spk or _strip(w_curr["word"]) in BACKCHANNEL_WORDS:
            continue
        gap_prev = w_curr["start_seconds"] - w_prev["end_seconds"]
        gap_next = w_next["start_seconds"] - w_curr["end_seconds"]
        if gap_prev < SMOOTHING_MAX_GAP_SECONDS and gap_next < SMOOTHING_MAX_GAP_SECONDS:
            smoothed[i] = (prev_spk, w_curr)
    return smoothed


def _build_turn(turn_id: int, speaker: str, words: List[Dict[str, Any]], split_reason: str) -> Dict[str, Any]:
    text = " ".join(w["word"] for w in words).strip()
    start = words[0]["start_seconds"]
    end = words[-1]["end_seconds"]
    return {
        "turn_id": turn_id,
        "speaker_label": speaker,
        "start_seconds": round(start, 3),
        "end_seconds": round(end, 3),
        "split_reason": split_reason,
        "is_short_turn": (end - start) < SHORT_TURN_MAX_SECONDS and _strip(text) in BACKCHANNEL_WORDS,
        "text": text,
        "words": words,
    }


def group_into_turns(
    assignments: List[Tuple[str, Dict[str, Any]]],
    pause_threshold: float = PAUSE_THRESHOLD_SECONDS,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Returns (turns, pause_splits, speaker_change_splits)."""
    turns: List[Dict[str, Any]] = []
    curr_spk: Optional[str] = None
    curr_words: List[Dict[str, Any]] = []
    last_word_end = 0.0
    pause_splits = 0
    speaker_splits = 0

    for spk, w in assignments:
        split_reason = None
        if curr_spk is not None and spk != curr_spk:
            split_reason = "speaker_change"
            speaker_splits += 1
        elif curr_spk is not None and (w["start_seconds"] - last_word_end) > pause_threshold:
            split_reason = "pause"
            pause_splits += 1

        if split_reason:
            turns.append(_build_turn(len(turns) + 1, curr_spk, curr_words, split_reason))
            curr_words = []

        curr_spk = spk
        curr_words.append(w)
        last_word_end = w["end_seconds"]

    if curr_words:
        turns.append(_build_turn(len(turns) + 1, curr_spk, curr_words, "end_of_audio"))

    return turns, pause_splits, speaker_splits


def _clean_word(w: Dict[str, Any]) -> Dict[str, Any]:
    confidence = w.get("confidence")
    return {
        "word": str(w["word"]).strip(),
        "start_seconds": round(float(w["start_seconds"]), 3),
        "end_seconds": round(float(w["end_seconds"]), 3),
        "confidence": round(float(confidence), 3) if confidence is not None else None,
        "timing_source": w["timing_source"],
    }


def reconcile_transcript(
    file_id: str,
    aligned_asr_path: str,
    diarization_path: str,
    output_path: str,
    audio_path: Optional[str] = None,
    pause_threshold: float = PAUSE_THRESHOLD_SECONDS,
) -> Dict[str, Any]:
    with open(aligned_asr_path, "r", encoding="utf-8") as f:
        asr_data = json.load(f)
    with open(diarization_path, "r", encoding="utf-8") as f:
        diar_data = json.load(f)

    intervals = diar_data.get("intervals", [])
    if not intervals:
        raise ValueError(f"No diarization intervals in {diarization_path}")

    all_words = [_clean_word(w) for seg in asr_data.get("segments", []) for w in seg.get("words", [])]
    if not all_words:
        raise ValueError(f"No aligned words found in {aligned_asr_path}")

    if audio_path and os.path.exists(audio_path):
        audio_duration = round(float(sf.info(audio_path).duration), 3)
    else:
        audio_duration = round(max(w["end_seconds"] for w in all_words), 3)

    assignments = [(assign_word_to_speaker(w, intervals), w) for w in all_words]
    assignments = smooth_isolated_words(assignments)
    turns, pause_splits, speaker_splits = group_into_turns(assignments, pause_threshold)

    telemetry = dict(asr_data.get("telemetry", {}))
    telemetry.update({
        "total_words": len(all_words),
        "total_turns": len(turns),
        "pause_splits": pause_splits,
        "speaker_change_splits": speaker_splits,
    })

    canonical_doc = {
        "file_id": file_id,
        "pipeline_version": PIPELINE_VERSION,
        "asr_model": asr_data["asr_model"],
        "alignment_model": asr_data["alignment_model"],
        "diarization_model": diar_data["diarizer_model"],
        "audio_duration_seconds": audio_duration,
        "upstream_cache_keys": {
            "align": asr_data.get("cache_key"),
            "diarize": diar_data.get("cache_key"),
        },
        "telemetry": telemetry,
        "speaker_labels": sorted({t["speaker_label"] for t in turns}),
        "turns": turns,
    }

    write_json(output_path, canonical_doc)
    print(f"[{file_id}] Reconciled {len(all_words)} words into {len(turns)} turns -> {output_path}")
    return canonical_doc


def run_batch_reconciliation(
    audio_files: List[str],
    asr_dir: str = RAW_ASR_DIR,
    diar_dir: str = DIARIZATION_DIR,
    out_dir: str = OUTPUT_DIR,
) -> List[str]:
    out_paths = []
    for audio_path in audio_files:
        base = file_base(audio_path)
        aligned_path = os.path.join(asr_dir, f"{base}_aligned.json")
        diar_path = os.path.join(diar_dir, f"{base}_diarization.json")
        for p in (aligned_path, diar_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"[{base}] Missing upstream artifact: {p}")
        out_path = os.path.join(out_dir, f"{base}_canonical.json")
        reconcile_transcript(os.path.basename(audio_path), aligned_path, diar_path, out_path, audio_path=audio_path)
        out_paths.append(out_path)
    return out_paths


if __name__ == "__main__":
    args = parse_stage_args("Stage 3: word-to-speaker turn reconciliation")
    ws = args.workspace
    run_batch_reconciliation(resolve_audio_files(args), asr_dir=ws.raw_asr_dir, diar_dir=ws.diarization_dir, out_dir=ws.output_dir)
