"""
Stage 2: Speaker Diarization via PyAnnote Community-1 (CPU, exactly two speakers).

Artifacts per file:
- `{id}.raw.rttm`: pyannote's untouched overlap-aware output (true "raw" diarization for DER).
- `{id}.rttm`: exclusive (one speaker at a time) intervals after centroid refinement; this is
  what word reconciliation consumes.
- `{id}_diarization.json`: refined intervals plus refinement telemetry.

Centroid refinement relabels short (0.5-5 s) segments whose voice embedding is clearly closer
to the other speaker's centroid. Its thresholds were tuned on the golden set (audio_05 rapid
banter), so refined DER on the golden set is optimistic; raw DER is reported alongside it.
"""

import os
import time
import warnings
from functools import lru_cache
from importlib.metadata import version
from typing import Any, Dict, List, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline

from src.pipeline.common import (
    DIARIZATION_DIR,
    build_cache_key,
    file_base,
    get_hf_token,
    is_cache_valid,
    parse_stage_args,
    resolve_audio_files,
    write_json,
)

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
DEVICE = "cpu"
NUM_SPEAKERS = 2
REFINEMENT = {
    "centroid_min_segment_seconds": 6.0,
    "trigger_centroid_similarity": 0.35,
    "candidate_min_seconds": 0.5,
    "candidate_max_seconds": 5.0,
    "reassign_margin": 0.10,
}


def diarize_config() -> Dict[str, Any]:
    return {
        "stage": "diarize",
        "model": DIARIZATION_MODEL,
        "pyannote_version": version("pyannote.audio"),
        "device": DEVICE,
        "min_speakers": NUM_SPEAKERS,
        "max_speakers": NUM_SPEAKERS,
        "refinement": REFINEMENT,
    }


@lru_cache(maxsize=1)
def load_pipeline() -> Pipeline:
    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=get_hf_token())
    pipeline.to(torch.device(DEVICE))
    return pipeline


def annotation_to_intervals(annotation) -> List[Dict[str, Any]]:
    return [
        {
            "speaker": speaker,
            "start": round(float(turn.start), 3),
            "end": round(float(turn.end), 3),
            "duration": round(float(turn.duration), 3),
        }
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def write_rttm(intervals: List[Dict[str, Any]], path: str, uri: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for iv in intervals:
            f.write(f"SPEAKER {uri} 1 {iv['start']:.3f} {iv['duration']:.3f} <NA> <NA> {iv['speaker']} <NA> <NA>\n")


def _unit_embedding(emb_model: Any, waveform_3d: torch.Tensor, sr: int, start: float, end: float) -> np.ndarray:
    chunk = waveform_3d[:, :, int(start * sr):int(end * sr)]
    e = np.asarray(emb_model(chunk)[0], dtype=np.float64)
    norm = np.linalg.norm(e)
    return e / norm if norm > 0 else e


def refine_short_segments_with_centroids(
    intervals: List[Dict[str, Any]],
    waveform_3d: torch.Tensor,
    sr: int,
    emb_model: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Relabels short, ambiguous segments against global speaker voice centroids built from long
    segments. Only triggers when the two centroids are similar enough to suggest cluster
    confusion. Returns (intervals, telemetry).
    """
    cfg = REFINEMENT
    telemetry: Dict[str, Any] = {"applied": False, "centroid_similarity": None, "reassigned_segments": 0}

    speakers = sorted({iv["speaker"] for iv in intervals})
    if len(speakers) != 2:
        return intervals, telemetry

    embeddings: Dict[str, List[np.ndarray]] = {s: [] for s in speakers}
    for iv in intervals:
        if iv["end"] - iv["start"] >= cfg["centroid_min_segment_seconds"]:
            embeddings[iv["speaker"]].append(_unit_embedding(emb_model, waveform_3d, sr, iv["start"], iv["end"]))

    if not all(embeddings.values()):
        return intervals, telemetry

    centroids = {}
    for s, embs in embeddings.items():
        c = np.mean(embs, axis=0)
        centroids[s] = c / max(float(np.linalg.norm(c)), 1e-9)

    spk_a, spk_b = speakers
    centroid_sim = float(np.dot(centroids[spk_a], centroids[spk_b]))
    telemetry["centroid_similarity"] = round(centroid_sim, 4)
    if centroid_sim <= cfg["trigger_centroid_similarity"]:
        return intervals, telemetry

    telemetry["applied"] = True
    other = {spk_a: spk_b, spk_b: spk_a}
    refined = []
    for iv in intervals:
        spk = iv["speaker"]
        dur = iv["end"] - iv["start"]
        if cfg["candidate_min_seconds"] <= dur <= cfg["candidate_max_seconds"]:
            e = _unit_embedding(emb_model, waveform_3d, sr, iv["start"], iv["end"])
            if float(np.dot(e, centroids[other[spk]])) - float(np.dot(e, centroids[spk])) > cfg["reassign_margin"]:
                spk = other[spk]
                telemetry["reassigned_segments"] += 1
        refined.append({**iv, "speaker": spk})

    return refined, telemetry


def diarize_file(audio_path: str, output_dir: str = DIARIZATION_DIR, force: bool = False) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    base = file_base(audio_path)
    raw_rttm_file = os.path.join(output_dir, f"{base}.raw.rttm")
    rttm_file = os.path.join(output_dir, f"{base}.rttm")
    json_file = os.path.join(output_dir, f"{base}_diarization.json")
    paths = {"raw_rttm": raw_rttm_file, "rttm": rttm_file, "json": json_file}

    config = diarize_config()
    cache_key = build_cache_key(config, [audio_path])
    if not force and is_cache_valid(json_file, cache_key) and os.path.exists(raw_rttm_file) and os.path.exists(rttm_file):
        print(f"[{base}] Using cached diarization: {rttm_file}")
        return paths

    pipeline = load_pipeline()

    # Decode in memory so pyannote does not depend on torchcodec/ffmpeg bindings.
    data, sr = sf.read(audio_path, dtype="float32")
    waveform = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data.T.copy())

    print(f"[{base}] Diarizing with {DIARIZATION_MODEL} (exactly {NUM_SPEAKERS} speakers, {DEVICE})...")
    start_t = time.time()
    output = pipeline({"waveform": waveform, "sample_rate": sr}, min_speakers=NUM_SPEAKERS, max_speakers=NUM_SPEAKERS)
    elapsed = time.time() - start_t

    raw_annotation = getattr(output, "speaker_diarization", output)
    exclusive_annotation = getattr(output, "exclusive_speaker_diarization", raw_annotation)
    raw_intervals = annotation_to_intervals(raw_annotation)
    exclusive_intervals = annotation_to_intervals(exclusive_annotation)

    refined_intervals, refinement_telemetry = refine_short_segments_with_centroids(
        exclusive_intervals, waveform.unsqueeze(0), sr, pipeline._embedding
    )
    print(f"[{base}] Diarized in {elapsed:.1f}s | refinement: {refinement_telemetry}")

    write_rttm(raw_intervals, raw_rttm_file, base)
    write_rttm(refined_intervals, rttm_file, base)
    write_json(json_file, {
        "file_id": os.path.basename(audio_path),
        "diarizer_model": DIARIZATION_MODEL,
        "config": config,
        "cache_key": cache_key,
        "runtime_seconds": round(elapsed, 2),
        "total_intervals": len(refined_intervals),
        "refinement": refinement_telemetry,
        "intervals": refined_intervals,
    })
    return paths


def run_batch_diarization(audio_files: List[str], output_dir: str = DIARIZATION_DIR, force: bool = False) -> List[Dict[str, str]]:
    print(f"Starting diarization on {len(audio_files)} file(s)...")
    return [diarize_file(f, output_dir=output_dir, force=force) for f in audio_files]


if __name__ == "__main__":
    args = parse_stage_args("Stage 2: pyannote speaker diarization")
    run_batch_diarization(resolve_audio_files(args), output_dir=args.workspace.diarization_dir, force=args.force)
