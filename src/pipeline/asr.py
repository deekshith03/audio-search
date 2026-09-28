"""
Stage 1A: ASR Transcription with faster-whisper (large-v3-turbo, fp32, CPU).

The same backend runs natively and inside Docker, so every environment produces identical
transcripts. fp32 was chosen over int8 after benchmarking: equal-or-better WER and faster on
ARM CPUs (see docs/PHASE_2_SUMMARY.md). Repetition loops are guarded with
condition_on_previous_text=False and ghost tokens in long silences with
hallucination_silence_threshold=2.0.
"""

import os
import time
from typing import Any, Dict, List, Optional

import faster_whisper
from faster_whisper import WhisperModel

from src.pipeline.common import (
    RAW_ASR_DIR,
    build_cache_key,
    file_base,
    is_cache_valid,
    parse_stage_args,
    resolve_audio_files,
    write_json,
)

MODEL_NAME = "large-v3-turbo"
MODEL_REPO = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
DEVICE = "cpu"
COMPUTE_TYPE = "float32"
DECODE_OPTIONS = {
    "language": "en",
    "beam_size": 5,
    "word_timestamps": True,
    "condition_on_previous_text": False,
    "hallucination_silence_threshold": 2.0,
    "vad_filter": False,
}


def asr_config() -> Dict[str, Any]:
    return {
        "stage": "asr",
        "backend": "faster-whisper",
        "backend_version": faster_whisper.__version__,
        "model": MODEL_REPO,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "decode_options": DECODE_OPTIONS,
    }


def load_model() -> WhisperModel:
    cpu_threads = int(os.environ.get("ASR_CPU_THREADS", os.cpu_count() or 4))
    return WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE, cpu_threads=cpu_threads)


def serialize_segments(segments) -> List[Dict[str, Any]]:
    serialized = []
    for seg in segments:
        serialized.append({
            "id": seg.id,
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "text": seg.text.strip(),
            "avg_logprob": round(float(seg.avg_logprob), 4),
            "no_speech_prob": round(float(seg.no_speech_prob), 4),
            "words": [
                {
                    "word": w.word.strip(),
                    "start": round(float(w.start), 3),
                    "end": round(float(w.end), 3),
                    "probability": round(float(w.probability), 4),
                }
                for w in (seg.words or [])
            ],
        })
    return serialized


def transcribe_file(
    audio_path: str,
    output_dir: str = RAW_ASR_DIR,
    force: bool = False,
    model: Optional[WhisperModel] = None,
) -> str:
    base = file_base(audio_path)
    out_file = os.path.join(output_dir, f"{base}_raw.json")
    config = asr_config()
    cache_key = build_cache_key(config, [audio_path])

    if not force and is_cache_valid(out_file, cache_key):
        print(f"[{base}] Using cached raw ASR: {out_file}")
        return out_file

    model = model or load_model()
    print(f"[{base}] Transcribing with faster-whisper {MODEL_NAME} ({COMPUTE_TYPE}, {DEVICE})...")
    start_t = time.time()
    segments_iter, info = model.transcribe(audio_path, **DECODE_OPTIONS)
    segments = serialize_segments(segments_iter)
    elapsed = time.time() - start_t
    print(f"[{base}] Completed ASR in {elapsed:.1f}s ({len(segments)} segments, RTF {elapsed / info.duration:.3f})")

    write_json(out_file, {
        "file_id": os.path.basename(audio_path),
        "model": MODEL_REPO,
        "config": config,
        "cache_key": cache_key,
        "runtime_seconds": round(elapsed, 2),
        "audio_duration_seconds": round(float(info.duration), 3),
        "language": info.language,
        "text": " ".join(s["text"] for s in segments).strip(),
        "segments": segments,
    })
    return out_file


def run_batch_asr(audio_files: List[str], force: bool = False) -> List[str]:
    print(f"Starting ASR on {len(audio_files)} file(s)...")
    model = None
    out_files = []
    for audio_path in audio_files:
        cache_key = build_cache_key(asr_config(), [audio_path])
        out_file = os.path.join(RAW_ASR_DIR, f"{file_base(audio_path)}_raw.json")
        if model is None and (force or not is_cache_valid(out_file, cache_key)):
            model = load_model()
        out_files.append(transcribe_file(audio_path, force=force, model=model))
    return out_files


if __name__ == "__main__":
    args = parse_stage_args("Stage 1A: faster-whisper ASR")
    run_batch_asr(resolve_audio_files(args), force=args.force)
