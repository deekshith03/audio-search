"""
Stage 2: Speaker Diarization via PyAnnote Community-1 on Apple Silicon.

Executes pyannote/speaker-diarization-community-1 on PyTorch MPS/CPU with k=2 constraint.
Uses in-memory waveform decoding to bypass torchcodec bindings.
Exports RTTM files for DER evaluation and exclusive intervals for word reconciliation.
"""

import os
import sys
import json
import time
import warnings
from typing import Dict, Any, List, Optional

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline


def load_env_file(env_path: str = ".env") -> None:
    """Loads key-value pairs from .env into os.environ if not already present."""
    if not os.path.exists(env_path):
        if env_path == ".env":
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            alt_path = os.path.join(repo_root, ".env")
            if os.path.exists(alt_path):
                env_path = alt_path
            else:
                return
        else:
            return

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k and k not in os.environ:
                os.environ[k] = v


def get_hf_token(env_path: Optional[str] = ".env") -> str:
    if env_path:
        load_env_file(env_path)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        # Check ~/.cache/huggingface/token
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_path):
            with open(token_path, "r") as f:
                token = f.read().strip()
    if not token:
        raise RuntimeError(
            "Hugging Face token not found. Please set HF_TOKEN or HUGGINGFACE_TOKEN "
            "in your .env file or environment, or log in via `huggingface-cli login`."
        )
    return token


def refine_short_segments_with_centroids(
    intervals: List[Dict[str, Any]],
    waveform: torch.Tensor,
    sr: int,
    emb_model: Any
) -> List[Dict[str, Any]]:
    """
    Refines short or ambiguous diarization segments by comparing against global speaker voice centroids.
    Only triggers when centroid similarity between the two speakers is high (>0.35), indicating
    potential cluster confusion in rapid banter (such as audio_05).
    """
    # 1. Compute Centroids from long, confident segments (>= 6.0s)
    e00 = []
    e01 = []
    # Ensure waveform has shape (batch, channels, samples)
    if waveform.ndim == 2:
        waveform_3d = waveform.unsqueeze(0)
    elif waveform.ndim == 1:
        waveform_3d = waveform.unsqueeze(0).unsqueeze(0)
    else:
        waveform_3d = waveform

    for iv in intervals:
        dur = iv["end"] - iv["start"]
        if dur >= 6.0:
            st_idx = int(iv["start"] * sr)
            et_idx = int(iv["end"] * sr)
            chunk = waveform_3d[:, :, st_idx:et_idx]
            e = emb_model(chunk)[0]
            norm = np.linalg.norm(e)
            if norm > 0:
                e = e / norm
                if iv["speaker"] == "SPEAKER_00":
                    e00.append(e)
                else:
                    e01.append(e)

    if not e00 or not e01:
        return intervals

    c00 = np.mean(e00, axis=0)
    c00 = c00 / max(float(np.linalg.norm(c00)), 1e-9)
    c01 = np.mean(e01, axis=0)
    c01 = c01 / max(float(np.linalg.norm(c01)), 1e-9)

    centroid_sim = float(np.dot(c00, c01))
    if centroid_sim <= 0.35:
        # Clusters are already well separated; no resegmentation needed
        return intervals

    resegmented = []
    reassigned = 0
    for iv in intervals:
        st, et, spk = iv["start"], iv["end"], iv["speaker"]
        dur = et - st
        if 0.5 <= dur <= 5.0:
            st_idx = int(st * sr)
            et_idx = int(et * sr)
            chunk = waveform_3d[:, :, st_idx:et_idx]
            e = emb_model(chunk)[0]
            norm = np.linalg.norm(e)
            if norm > 0:
                e = e / norm
                s00 = float(np.dot(e, c00))
                s01 = float(np.dot(e, c01))
                if spk == "SPEAKER_01" and (s00 - s01) > 0.10:
                    spk = "SPEAKER_00"
                    reassigned += 1
                elif spk == "SPEAKER_00" and (s01 - s00) > 0.10:
                    spk = "SPEAKER_01"
                    reassigned += 1

        resegmented.append({
            "speaker": spk,
            "start": round(st, 3),
            "end": round(et, 3),
            "duration": round(dur, 3)
        })

    if reassigned > 0:
        print(f"  [Centroid Refinement] Reassigned {reassigned} short segments based on acoustic voice centroids (centroid sim: {centroid_sim:.3f})")

    return resegmented


def diarize_file(
    audio_path: str,
    output_dir: str = "dataset/cache/raw_diarization",
    device_name: str = "mps",
    force: bool = False
) -> Dict[str, str]:
    """
    Diarizes a 16kHz mono WAV file using PyAnnote Community-1.
    Saves RTTM and JSON intervals.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_id = os.path.basename(audio_path)
    base_name = file_id.replace(".wav", "")
    rttm_file = os.path.join(output_dir, f"{base_name}.rttm")
    json_file = os.path.join(output_dir, f"{base_name}_diarization.json")

    if not force and os.path.exists(rttm_file) and os.path.exists(json_file):
        print(f"[{file_id}] Using cached diarization: {rttm_file}")
        return {"rttm": rttm_file, "json": json_file}

    token = get_hf_token()
    print(f"[{file_id}] Loading pyannote community-1 pipeline ({device_name})...")
    start_t = time.time()

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token
    )

    device = torch.device(device_name if torch.backends.mps.is_available() and device_name == "mps" else "cpu")
    pipeline.to(device)

    # In-memory preloaded audio decoding
    data, sr = sf.read(audio_path)
    if data.ndim == 1:
        waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    else:
        waveform = torch.tensor(data.T, dtype=torch.float32)

    audio_dict = {"waveform": waveform, "sample_rate": sr}
    
    print(f"[{file_id}] Running diarization with min_speakers=2, max_speakers=2...")
    output = pipeline(audio_dict, min_speakers=2, max_speakers=2)
    elapsed = time.time() - start_t
    print(f"[{file_id}] Completed diarization in {elapsed:.1f}s")

    # 1. Write RTTM file for standard DER evaluation (multi-track)
    annotation = getattr(output, "speaker_diarization", output)
    with open(rttm_file, "w", encoding="utf-8") as f:
        annotation.write_rttm(f)

    # 2. Extract exclusive intervals for word reconciliation
    exclusive_annotation = getattr(output, "exclusive_speaker_diarization", annotation)
    intervals = []
    for turn, _, speaker in exclusive_annotation.itertracks(yield_label=True):
        intervals.append({
            "speaker": speaker,
            "start": round(float(turn.start), 3),
            "end": round(float(turn.end), 3),
            "duration": round(float(turn.duration), 3)
        })

    # 3. Apply Acoustic Voice Centroid Refinement for fast banter / short interjections
    if hasattr(pipeline, "_embedding"):
        intervals = refine_short_segments_with_centroids(
            intervals,
            waveform.unsqueeze(0) if waveform.ndim == 2 else waveform,
            sr,
            pipeline._embedding
        )

    # 4. Write refined RTTM file
    with open(rttm_file, "w", encoding="utf-8") as f:
        for iv in intervals:
            f.write(f"SPEAKER {file_id.replace('.wav', '')} 1 {iv['start']:.3f} {iv['duration']:.3f} <NA> <NA> {iv['speaker']} <NA> <NA>\n")

    json_data = {
        "file_id": file_id,
        "diarizer_model": "pyannote/speaker-diarization-community-1",
        "runtime_seconds": round(elapsed, 2),
        "total_intervals": len(intervals),
        "intervals": intervals
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    return {"rttm": rttm_file, "json": json_file}


def run_batch_diarization(audio_dir: str = "dataset/audio") -> List[Dict[str, str]]:
    """Diarizes all WAV files in the audio directory."""
    files = sorted([os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")])
    results = []
    print(f"Starting Batch Diarization on {len(files)} files...")
    for f in files:
        try:
            res = diarize_file(f, device_name="mps")
            results.append(res)
        except Exception as e:
            print(f"[{os.path.basename(f)}] MPS diarization failed ({e}), falling back to CPU...")
            res = diarize_file(f, device_name="cpu")
            results.append(res)
    return results


if __name__ == "__main__":
    run_batch_diarization()
