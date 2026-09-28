"""
Stage 3: Deterministic Word-to-Speaker Turn Reconciliation & Golden Enrichment.

Implements strict reconciliation rules:
1. Word Assignment: Assigns each word to the speaker whose interval overlaps it by strictly >50%.
   If overlap <= 50% or word falls in a gap, falls back to the temporally nearest speaker interval.
2. Gap Handling: Words falling in unassigned gaps inherit the nearest speaker segment in time.
3. Turn Splitting:
   - Speaker change -> split_reason: "speaker_change"
   - Natural pause > 1.5s -> split_reason: "pause"
   - End of audio -> split_reason: "end_of_audio"
4. Short Turn Tagging: Turns < 0.8s matching backchannel lexicons tagged as is_short_turn=True.
5. Golden Set Enrichment: Maps SPEAKER_00 / SPEAKER_01 to real names via globally optimal
   bipartite overlap against reference ground truth.
6. Schema Conformance: Emits audio_duration_seconds, start_seconds, end_seconds, confidence,
   and standard telemetry matching PHASE_2_SPECIFICATION.md.
"""

import os
import sys
import json
import itertools
from typing import Dict, Any, List, Optional
import soundfile as sf


BACKCHANNEL_WORDS = {"yeah", "right", "okay", "sure", "yep", "mm-hmm", "uh-huh", "yes", "i see", "exactly"}


def assign_word_to_speaker(word: Dict[str, Any], intervals: List[Dict[str, Any]]) -> str:
    """
    Assigns word to speaker with strictly >50% overlap.
    If overlap <= 50% or word is in a diarization gap, falls back to temporally nearest interval.
    """
    w_start = word.get("start_seconds", word.get("start", 0.0))
    w_end = word.get("end_seconds", word.get("end", 0.0))
    w_dur = max(w_end - w_start, 0.001)

    spk_overlaps: Dict[str, float] = {}
    for iv in intervals:
        s_start = iv["start"]
        s_end = iv["end"]
        overlap = max(0.0, min(w_end, s_end) - max(w_start, s_start))
        if overlap > 0:
            spk_overlaps[iv["speaker"]] = spk_overlaps.get(iv["speaker"], 0.0) + overlap

    if spk_overlaps:
        best_spk, max_overlap = max(spk_overlaps.items(), key=lambda x: x[1])
        if (max_overlap / w_dur) > 0.5:
            return best_spk

    # Gap / low-overlap fallback: find temporally nearest interval
    nearest_spk = intervals[0]["speaker"] if intervals else "SPEAKER_00"
    min_dist = float("inf")

    for iv in intervals:
        s_start = iv["start"]
        s_end = iv["end"]
        if w_end < s_start:
            dist = s_start - w_end
        elif w_start > s_end:
            dist = w_start - s_end
        else:
            dist = 0.0

        if dist < min_dist:
            min_dist = dist
            nearest_spk = iv["speaker"]

    return nearest_spk


def compute_speaker_mapping(
    pipeline_turns: List[Dict[str, Any]],
    reference_path: str
) -> Dict[str, str]:
    """
    Computes globally optimal bipartite mapping between raw pipeline labels (SPEAKER_00/01)
    and ground-truth human names via maximal temporal overlap across all permutations.
    """
    if not os.path.exists(reference_path):
        return {}

    with open(reference_path, "r", encoding="utf-8") as f:
        ref_data = json.load(f)

    ref_turns = ref_data.get("turns", [])
    if not ref_turns:
        return {}

    p_speakers = sorted(list({pt["speaker_label"] for pt in pipeline_turns}))
    r_speakers = sorted(list({rt["speaker"] for rt in ref_turns}))

    if not p_speakers or not r_speakers:
        return {}

    overlap_matrix: Dict[str, Dict[str, float]] = {p: {r: 0.0 for r in r_speakers} for p in p_speakers}

    for pt in pipeline_turns:
        p_spk = pt["speaker_label"]
        p_st = pt["start_seconds"]
        p_et = pt["end_seconds"]

        for rt in ref_turns:
            r_spk = rt["speaker"]
            r_st = rt["start_time"]
            r_et = rt["end_time"]

            overlap = max(0.0, min(p_et, r_et) - max(p_st, r_st))
            if overlap > 0:
                overlap_matrix[p_spk][r_spk] += overlap

    # Globally optimal assignment by evaluating all bijective permutations
    best_mapping: Dict[str, str] = {}
    best_score = -1.0

    # Ensure reference speakers pool can cover pipeline speakers
    if len(r_speakers) >= len(p_speakers):
        for perm in itertools.permutations(r_speakers, len(p_speakers)):
            current_mapping = dict(zip(p_speakers, perm))
            score = sum(overlap_matrix[p][r] for p, r in current_mapping.items())
            if score > best_score:
                best_score = score
                best_mapping = current_mapping
    else:
        # Fallback if fewer reference speakers than pipeline speakers
        for p in p_speakers:
            best_ref = max(overlap_matrix[p].items(), key=lambda x: x[1])[0]
            best_mapping[p] = best_ref

    return best_mapping


def reconcile_transcript(
    file_id: str,
    aligned_asr_path: str,
    diarization_path: str,
    output_path: str,
    reference_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    pause_threshold: float = 1.5
) -> Dict[str, Any]:
    """
    Reconciles word timestamps and speaker intervals into canonical turn-by-turn transcript
    strictly complying with docs/PHASE_2_SPECIFICATION.md schema.
    Emits native anonymous speaker labels ("Speaker 1", "Speaker 2") with zero ground-truth leakage.
    """
    with open(aligned_asr_path, "r", encoding="utf-8") as f:
        asr_data = json.load(f)
    with open(diarization_path, "r", encoding="utf-8") as f:
        diar_data = json.load(f)

    intervals = diar_data.get("intervals", [])
    if not intervals:
        raise ValueError(f"No diarization intervals in {diarization_path}")

    # Extract all words from aligned segments
    all_words = []
    for seg in asr_data.get("segments", []):
        all_words.extend(seg.get("words", []))

    if not all_words:
        raise ValueError(f"No aligned words found in {aligned_asr_path}")

    # Determine audio duration directly from audio file (or fallback to max word timestamp)
    audio_duration = 0.0
    if audio_path and os.path.exists(audio_path):
        try:
            info = sf.info(audio_path)
            audio_duration = round(float(info.duration), 2)
        except Exception:
            pass

    if audio_duration == 0.0:
        audio_duration = round(max(float(w.get("end_seconds", w.get("end", 0.0))) for w in all_words), 2)

    # Step 1: Assign each word to a speaker with strict >50% overlap rule
    word_assignments = []
    for w in all_words:
        spk = assign_word_to_speaker(w, intervals)
        w_start = round(float(w.get("start_seconds", w.get("start", 0.0))), 3)
        w_end = round(float(w.get("end_seconds", w.get("end", 0.0))), 3)
        confidence = round(float(w.get("confidence", w.get("score", 1.0))), 3)
        timing_source = w.get("timing_source", "wav2vec2_aligned")

        clean_word = {
            "word": str(w["word"]).strip(),
            "start_seconds": w_start,
            "end_seconds": w_end,
            "confidence": confidence,
            "timing_source": timing_source
        }
        word_assignments.append((spk, clean_word))

    # Step 1.5: Sequence smoothing - collapse isolated 1-word intra-sentence glitches (<0.4s pause)
    smoothed_assignments = list(word_assignments)
    for i in range(1, len(word_assignments) - 1):
        prev_spk, w_prev = smoothed_assignments[i - 1]
        curr_spk, w_curr = smoothed_assignments[i]
        next_spk, w_next = smoothed_assignments[i + 1]

        w_text = w_curr["word"].lower().strip(".,!?;:\"'")
        if prev_spk == next_spk and curr_spk != prev_spk:
            gap_prev = w_curr["start_seconds"] - w_prev["end_seconds"]
            gap_next = w_next["start_seconds"] - w_curr["end_seconds"]
            if gap_prev < 0.4 and gap_next < 0.4 and w_text not in BACKCHANNEL_WORDS:
                smoothed_assignments[i] = (prev_spk, w_curr)

    word_assignments = smoothed_assignments

    # Step 2: Group consecutive words into turns
    turns = []
    curr_spk = None
    curr_turn_words = []
    last_word_end = 0.0
    pause_splits = 0
    speaker_splits = 0

    for spk, w in word_assignments:
        w_start = w["start_seconds"]
        w_end = w["end_seconds"]

        split_reason = None
        if curr_spk is None:
            curr_spk = spk
        elif spk != curr_spk:
            split_reason = "speaker_change"
            speaker_splits += 1
        elif (w_start - last_word_end) > pause_threshold:
            split_reason = "pause"
            pause_splits += 1

        if split_reason:
            t_text = " ".join(cw["word"] for cw in curr_turn_words).strip()
            t_start = curr_turn_words[0]["start_seconds"]
            t_end = curr_turn_words[-1]["end_seconds"]
            t_dur = t_end - t_start
            is_short = (t_dur < 0.8) and (t_text.lower().strip(".,!?;:\"'") in BACKCHANNEL_WORDS)

            turns.append({
                "turn_id": len(turns) + 1,
                "speaker_label": curr_spk,
                "start_seconds": round(t_start, 3),
                "end_seconds": round(t_end, 3),
                "split_reason": split_reason,
                "is_short_turn": is_short,
                "text": t_text,
                "words": curr_turn_words
            })

            curr_spk = spk
            curr_turn_words = []

        curr_turn_words.append(w)
        last_word_end = w_end

    # Append final turn with split_reason: "end_of_audio"
    if curr_turn_words:
        t_text = " ".join(cw["word"] for cw in curr_turn_words).strip()
        t_start = curr_turn_words[0]["start_seconds"]
        t_end = curr_turn_words[-1]["end_seconds"]
        t_dur = t_end - t_start
        is_short = (t_dur < 0.8) and (t_text.lower().strip(".,!?;:\"'") in BACKCHANNEL_WORDS)

        turns.append({
            "turn_id": len(turns) + 1,
            "speaker_label": curr_spk,
            "start_seconds": round(t_start, 3),
            "end_seconds": round(t_end, 3),
            "split_reason": "end_of_audio",
            "is_short_turn": is_short,
            "text": t_text,
            "words": curr_turn_words
        })

    # Step 3: Zero-shot Anonymous Speaker Mapping (Speaker 1, Speaker 2)
    unique_spks = sorted(list({t["speaker_label"] for t in turns}))
    mapping = {spk_lbl: f"Speaker {idx + 1}" for idx, spk_lbl in enumerate(unique_spks)}

    ordered_turns = []
    for t in turns:
        spk_name = mapping.get(t["speaker_label"], t["speaker_label"])
        ordered_turn = {
            "turn_id": t["turn_id"],
            "speaker_label": t["speaker_label"],
            "speaker_name": spk_name,
            "start_seconds": t["start_seconds"],
            "end_seconds": t["end_seconds"],
            "split_reason": t["split_reason"],
            "is_short_turn": t["is_short_turn"],
            "text": t["text"],
            "words": t["words"]
        }
        ordered_turns.append(ordered_turn)

    raw_telem = asr_data.get("telemetry", {})
    fallback_words = raw_telem.get("fallback_aligned_words", raw_telem.get("fallback_words", 0))
    fallback_rate = raw_telem.get("alignment_fallback_rate", raw_telem.get("fallback_rate", 0.0))

    telemetry = {
        "total_words": len(all_words),
        "fallback_aligned_words": fallback_words,
        "alignment_fallback_rate": round(float(fallback_rate), 4),
        "total_turns": len(ordered_turns),
        "pause_splits": pause_splits,
        "speaker_change_splits": speaker_splits
    }

    canonical_doc = {
        "file_id": file_id,
        "pipeline_version": "2.0.0",
        "asr_model": asr_data.get("model", "mlx-community/whisper-large-v3-turbo"),
        "diarization_model": diar_data.get("diarizer_model", "pyannote/speaker-diarization-community-1"),
        "alignment_model": asr_data.get("alignment_model", "wav2vec2-large-960h"),
        "audio_duration_seconds": audio_duration,
        "telemetry": telemetry,
        "speaker_mapping": mapping,
        "turns": ordered_turns
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(canonical_doc, f, indent=2, ensure_ascii=False)

    print(f"[{file_id}] Reconciled {len(all_words)} words into {len(ordered_turns)} canonical turns -> {output_path}")
    print(f"[{file_id}] Anonymous Speaker mapping: {mapping}")

    return canonical_doc


def run_batch_reconciliation(
    audio_dir: str = "dataset/audio",
    asr_dir: str = "dataset/cache/raw_asr",
    diar_dir: str = "dataset/cache/raw_diarization",
    gt_dir: str = "dataset/ground_truth",
    out_dir: str = "dataset/pipeline_outputs"
):
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
    for af in audio_files:
        base = af.replace(".wav", "")
        audio_path = os.path.join(audio_dir, af)
        aligned_path = os.path.join(asr_dir, f"{base}_aligned.json")
        diar_path = os.path.join(diar_dir, f"{base}_diarization.json")
        out_path = os.path.join(out_dir, f"{base}_canonical.json")

        if not os.path.exists(aligned_path) or not os.path.exists(diar_path):
            print(f"[{af}] Skipping: Missing aligned ASR ({os.path.exists(aligned_path)}) or Diarization ({os.path.exists(diar_path)})")
            continue

        reconcile_transcript(af, aligned_path, diar_path, out_path, audio_path=audio_path)


if __name__ == "__main__":
    run_batch_reconciliation()
