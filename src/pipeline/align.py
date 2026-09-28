"""
Stage 1B: Word-Level Forced Alignment via Wav2Vec2 (WhisperX Aligner).

Refines Whisper's cross-attention timestamps down to frame-level (~20 ms)
phoneme boundaries. Implements linear interpolation fallback for untimed
symbol and digit tokens ("$5", "2026", "N3", "%") and tracks fallback telemetry.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple

import torch
import whisperx


def align_file(
    audio_path: str,
    raw_asr_path: str,
    output_dir: str = "dataset/cache/raw_asr",
    device: str = "cpu"
) -> str:
    """
    Performs forced alignment on a raw ASR transcript and saves aligned word cache.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_id = os.path.basename(audio_path)
    base_name = file_id.replace(".wav", "")
    out_file = os.path.join(output_dir, f"{base_name}_aligned.json")

    if os.path.exists(out_file):
        print(f"[{file_id}] Using cached aligned ASR: {out_file}")
        return out_file

    with open(raw_asr_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_segments = raw_data.get("segments", [])
    if not raw_segments:
        print(f"[{file_id}] Warning: No segments found in {raw_asr_path}")
        return out_file

    print(f"[{file_id}] Loading Wav2Vec2 alignment model ({device})...")
    start_t = time.time()
    
    align_model, align_metadata = whisperx.load_align_model(
        language_code="en",
        device=device
    )

    audio = whisperx.load_audio(audio_path)
    
    print(f"[{file_id}] Running forced alignment over {len(raw_segments)} segments...")
    aligned_result = whisperx.align(
        raw_segments,
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False
    )
    
    elapsed = time.time() - start_t
    print(f"[{file_id}] Alignment complete in {elapsed:.1f}s")

    # Post-process: interpolate untimed tokens (symbols, digits, acronyms)
    total_words = 0
    fallback_words = 0
    processed_segments = []

    for seg in aligned_result.get("segments", []):
        words = seg.get("words", [])
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start + 1.0))
        seg_words = []

        for w_idx, w in enumerate(words):
            total_words += 1
            word_str = w.get("word", "").strip()
            w_start = w.get("start")
            w_end = w.get("end")
            score = w.get("score", 1.0)

            # Untimed token handling: interpolate or fall back
            if w_start is None or w_end is None:
                fallback_words += 1
                # Find previous timed word end
                prev_end = seg_start
                for p in reversed(seg_words):
                    if p.get("end") is not None:
                        prev_end = p["end"]
                        break
                
                # Find next timed word start
                next_start = seg_end
                for n in words[w_idx + 1:]:
                    if n.get("start") is not None:
                        next_start = n["start"]
                        break

                w_start = round(prev_end, 3)
                w_end = round(max(prev_end + 0.15, min(next_start, prev_end + 0.35)), 3)
                timing_src = "interpolated_fallback"
            else:
                timing_src = "wav2vec2_aligned"

            w_start_val = round(float(w_start), 3)
            w_end_val = round(float(w_end), 3)
            w_score_val = round(float(score), 3) if score is not None else 1.0

            seg_words.append({
                "word": word_str,
                "start_seconds": w_start_val,
                "end_seconds": w_end_val,
                "confidence": w_score_val,
                "timing_source": timing_src
            })

        processed_segments.append({
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "text": seg.get("text", "").strip(),
            "words": seg_words
        })

    fallback_rate = (fallback_words / total_words) if total_words > 0 else 0.0
    print(f"[{file_id}] Total words: {total_words} | Untimed fallback words: {fallback_words} ({fallback_rate:.2%})")

    out_data = {
        "file_id": file_id,
        "alignment_model": "wav2vec2-large-960h",
        "runtime_seconds": round(elapsed, 2),
        "telemetry": {
            "total_words": total_words,
            "fallback_aligned_words": fallback_words,
            "alignment_fallback_rate": round(fallback_rate, 4)
        },
        "segments": processed_segments
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    return out_file


def run_batch_alignment(audio_dir: str = "dataset/audio", asr_dir: str = "dataset/cache/raw_asr") -> List[str]:
    """Runs forced alignment across all audio files."""
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
    aligned_files = []
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Starting Batch Alignment on {len(audio_files)} files using {device}...")

    for af in audio_files:
        audio_path = os.path.join(audio_dir, af)
        raw_asr_path = os.path.join(asr_dir, af.replace(".wav", "_raw.json"))
        if not os.path.exists(raw_asr_path):
            print(f"[{af}] Raw ASR missing: {raw_asr_path}. Run asr.py first.")
            continue
        try:
            out = align_file(audio_path, raw_asr_path, output_dir=asr_dir, device=device)
            aligned_files.append(out)
        except Exception as e:
            print(f"[{af}] MPS alignment failed ({e}), falling back to CPU...")
            out = align_file(audio_path, raw_asr_path, output_dir=asr_dir, device="cpu")
            aligned_files.append(out)

    return aligned_files


if __name__ == "__main__":
    run_batch_alignment()
