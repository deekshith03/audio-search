"""
Stage 0: Audio Ingestion & Normalization.

Validates an arbitrary user upload with ffprobe and converts it with ffmpeg to the format every
downstream stage assumes: 16 kHz, mono, 16-bit PCM WAV. Uploads get a content-addressed
`file_id` ("upload_" + SHA-256 of the original bytes + ".wav", the same "<name>.wav" convention as
the golden set), so re-uploading the same recording reuses its results.
"""

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Optional

from src.pipeline.common import AUDIO_DIR, file_base, sha256_file, write_json

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm", ".mp4"}
TARGET_SAMPLE_RATE = 16000
MIN_DURATION_SECONDS = 10.0
MAX_DURATION_SECONDS = float(os.environ.get("MAX_DURATION_SECONDS", 600))


class IngestError(ValueError):
    pass


@dataclass
class IngestResult:
    file_id: str
    wav_path: str
    source_filename: str
    source_sha256: str
    source_format: str
    source_codec: str
    source_sample_rate: int
    source_channels: int
    duration_seconds: float
    reused_existing: bool


def _require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise IngestError(f"`{tool}` is not installed or not on PATH; it is required to decode uploads.")


def format_duration(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def probe(path: str) -> dict:
    _require_ffmpeg()
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise IngestError(f"Could not decode file (ffprobe: {proc.stderr.strip() or 'unknown error'}).")
    info = json.loads(proc.stdout or "{}")
    audio_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise IngestError("File contains no audio stream.")

    stream = audio_streams[0]
    duration = stream.get("duration") or info.get("format", {}).get("duration")
    if duration is None:
        raise IngestError("Could not determine audio duration.")
    return {
        "format": info.get("format", {}).get("format_name", "unknown"),
        "codec": stream.get("codec_name", "unknown"),
        "sample_rate": int(stream.get("sample_rate", 0) or 0),
        "channels": int(stream.get("channels", 0) or 0),
        "duration": float(duration),
    }


def validate_upload(path: str, max_duration: float = MAX_DURATION_SECONDS, min_duration: float = MIN_DURATION_SECONDS) -> dict:
    if not os.path.exists(path):
        raise IngestError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestError(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.")
    if os.path.getsize(path) == 0:
        raise IngestError("File is empty.")

    info = probe(path)
    if info["duration"] > max_duration:
        raise IngestError(f"Audio too long ({format_duration(info['duration'])} > {format_duration(max_duration)}).")
    if info["duration"] < min_duration:
        raise IngestError(f"Audio too short ({info['duration']:.1f}s < {min_duration:.0f}s).")
    return info


def normalize_to_wav(src_path: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    tmp_path = f"{dst_path}.tmp.wav"
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", src_path,
            "-map", "0:a:0", "-vn",
            "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), "-sample_fmt", "s16", "-c:a", "pcm_s16le",
            tmp_path,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise IngestError(f"Audio conversion failed (ffmpeg: {proc.stderr.strip() or 'unknown error'}).")
    os.replace(tmp_path, dst_path)


def ingest(
    src_path: str,
    out_dir: str = AUDIO_DIR,
    file_id: Optional[str] = None,
    max_duration: float = MAX_DURATION_SECONDS,
) -> IngestResult:
    info = validate_upload(src_path, max_duration=max_duration)
    source_sha = sha256_file(src_path)
    base = file_base(file_id) if file_id else f"upload_{source_sha[:16]}"
    file_id = f"{base}.wav"
    wav_path = os.path.join(out_dir, file_id)
    meta_path = os.path.join(out_dir, f"{base}.ingest.json")

    reused = os.path.exists(wav_path) and os.path.exists(meta_path)
    if reused:
        with open(meta_path, "r", encoding="utf-8") as f:
            reused = json.load(f).get("source_sha256") == source_sha

    if not reused:
        normalize_to_wav(src_path, wav_path)

    result = IngestResult(
        file_id=file_id,
        wav_path=wav_path,
        source_filename=os.path.basename(src_path),
        source_sha256=source_sha,
        source_format=info["format"],
        source_codec=info["codec"],
        source_sample_rate=info["sample_rate"],
        source_channels=info["channels"],
        duration_seconds=round(info["duration"], 3),
        reused_existing=reused,
    )
    if not reused:
        write_json(meta_path, asdict(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 0: validate and normalize an audio file to 16 kHz mono WAV")
    parser.add_argument("source")
    parser.add_argument("--out-dir", default=AUDIO_DIR)
    parser.add_argument("--file-id", default=None)
    args = parser.parse_args()
    try:
        res = ingest(args.source, out_dir=args.out_dir, file_id=args.file_id)
    except IngestError as e:
        raise SystemExit(f"Ingest rejected: {e}")
    print(json.dumps(asdict(res), indent=2))
