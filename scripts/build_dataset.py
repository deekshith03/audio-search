"""
Rebuilds or verifies the golden audio set from dataset/metadata/sources.json.

  python -m scripts.build_dataset --verify   # check committed WAVs match recorded format + SHA-256
  python -m scripts.build_dataset --build    # clip, normalize and (optionally) noise-mix from sources

Build steps per file:
1. ffmpeg cuts [clip_start_seconds, clip_end_seconds] from source_audio_url (URL or local path)
   and converts it to 16 kHz mono 16-bit PCM.
2. If `noise_mix` is set, the noise profile is looped/trimmed to the clip length and scaled so
   that RMS(speech) / RMS(noise) equals the target SNR, then summed and peak-limited to [-1, 1].

Entries with missing provenance (null fields) are reported and skipped; the command exits
non-zero so an incomplete recipe is never mistaken for a reproducible one. A rebuilt file may
differ byte-for-byte from the committed one if the upstream audio or encoder changed, so the
SHA-256 comparison is reported rather than silently accepted.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

import numpy as np
import soundfile as sf

from src.pipeline.common import AUDIO_DIR, sha256_file

SOURCES_PATH = "dataset/metadata/sources.json"
REQUIRED_BUILD_FIELDS = ("source_audio_url", "clip_start_seconds", "clip_end_seconds")


def load_sources() -> Dict[str, Any]:
    with open(SOURCES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def verify(entries: List[Dict[str, Any]]) -> int:
    failures = 0
    for e in entries:
        path = os.path.join(AUDIO_DIR, e["file_id"])
        expected = e["output"]
        if not os.path.exists(path):
            print(f"✗ {e['file_id']}: missing")
            failures += 1
            continue
        info = sf.info(path)
        problems = []
        if (info.samplerate, info.channels, info.subtype) != (expected["sample_rate"], expected["channels"], expected["subtype"]):
            problems.append(f"format {info.samplerate}/{info.channels}/{info.subtype}")
        if sha256_file(path) != expected["sha256"]:
            problems.append("sha256 mismatch")
        print(f"{'✗' if problems else '✓'} {e['file_id']}{': ' + ', '.join(problems) if problems else ''}")
        failures += bool(problems)
    return failures


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    noise = noise[:len(speech)]
    noise_rms = rms(noise)
    if noise_rms == 0:
        raise ValueError("Noise profile is silent.")
    gain = rms(speech) / (noise_rms * 10 ** (snr_db / 20))
    return np.clip(speech + gain * noise, -1.0, 1.0)


def clip_and_normalize(src: str, start: float, end: float, dst: str) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", str(start), "-to", str(end), "-i", src,
         "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst],
        check=True,
    )


def build(entries: List[Dict[str, Any]], out_dir: str) -> int:
    incomplete = 0
    os.makedirs(out_dir, exist_ok=True)
    for e in entries:
        missing = [k for k in REQUIRED_BUILD_FIELDS if e.get(k) is None]
        if missing:
            print(f"- {e['file_id']}: skipped, missing provenance fields {missing}")
            incomplete += 1
            continue

        dst = os.path.join(out_dir, e["file_id"])
        with tempfile.TemporaryDirectory() as tmp:
            clean = os.path.join(tmp, "clean.wav")
            clip_and_normalize(e["source_audio_url"], e["clip_start_seconds"], e["clip_end_seconds"], clean)
            if e.get("noise_mix"):
                speech, sr = sf.read(clean, dtype="float32")
                noise, noise_sr = sf.read(e["noise_mix"]["noise_file"], dtype="float32")
                if noise_sr != sr:
                    raise ValueError(f"Noise sample rate {noise_sr} != {sr}")
                sf.write(dst, mix_at_snr(speech, noise, e["noise_mix"]["snr_db"]), sr, subtype="PCM_16")
            else:
                os.replace(clean, dst)

        match = sha256_file(dst) == e["output"]["sha256"]
        print(f"+ {e['file_id']}: built ({'sha256 matches' if match else 'sha256 differs from committed file'})")
    return incomplete


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify or rebuild the golden audio set")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify", action="store_true")
    group.add_argument("--build", action="store_true")
    parser.add_argument("--out-dir", default="dataset/rebuilt_audio", help="Where --build writes files (never overwrites dataset/audio).")
    args = parser.parse_args()

    entries = load_sources()["files"]
    failures = verify(entries) if args.verify else build(entries, args.out_dir)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
