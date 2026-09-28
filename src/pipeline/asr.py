"""
Stage 1A: Batch ASR Transcription on Apple Silicon Metal GPU.

Uses mlx-whisper running whisper-large-v3-turbo directly on the M-series GPU.
Guards against repetition loops with condition_on_previous_text=False
and suppresses ghost tokens with hallucination_silence_threshold=2.0.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List

import mlx_whisper


def transcribe_file(
    audio_path: str,
    output_dir: str = "dataset/cache/raw_asr",
    model_name: str = "mlx-community/whisper-large-v3-turbo"
) -> str:
    """
    Transcribes an audio file on Apple Metal GPU and saves raw JSON cache.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_id = os.path.basename(audio_path)
    base_name = file_id.replace(".wav", "")
    out_file = os.path.join(output_dir, f"{base_name}_raw.json")

    if os.path.exists(out_file):
        print(f"[{file_id}] Using cached raw ASR: {out_file}")
        return out_file

    print(f"[{file_id}] Transcribing on Apple Metal GPU with {model_name}...")
    start_t = time.time()
    
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model_name,
        word_timestamps=True,
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0
    )
    elapsed = time.time() - start_t
    print(f"[{file_id}] Completed ASR in {elapsed:.1f}s ({len(result.get('segments', []))} segments)")

    cache_data = {
        "file_id": file_id,
        "model": model_name,
        "runtime_seconds": round(elapsed, 2),
        "text": result.get("text", "").strip(),
        "segments": result.get("segments", [])
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)

    return out_file


def run_batch_asr(audio_dir: str = "dataset/audio") -> List[str]:
    """Transcribes all WAV files in the audio directory."""
    files = sorted([os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")])
    out_files = []
    print(f"Starting Batch ASR on {len(files)} files...")
    for f in files:
        out = transcribe_file(f)
        out_files.append(out)
    return out_files


if __name__ == "__main__":
    run_batch_asr()
