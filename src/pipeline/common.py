"""
Shared helpers for pipeline stages: cache fingerprints, per-file CLI, and Hugging Face auth.

Every stage artifact embeds a `cache_key` built from the stage config plus the SHA-256 of its
inputs. A cached artifact is reused only when its key matches the current one, so changing a
model, a parameter, or an upstream artifact invalidates everything downstream automatically.
"""

import argparse
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

AUDIO_DIR = "dataset/audio"
RAW_ASR_DIR = "dataset/cache/raw_asr"
DIARIZATION_DIR = "dataset/cache/raw_diarization"
OUTPUT_DIR = "dataset/pipeline_outputs"
GROUND_TRUTH_DIR = "dataset/ground_truth"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def build_cache_key(config: Dict[str, Any], input_paths: List[str]) -> str:
    payload = {
        "config": config,
        "inputs": {os.path.basename(p): sha256_file(p) for p in input_paths},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def is_cache_valid(artifact_path: str, cache_key: str) -> bool:
    if not os.path.exists(artifact_path):
        return False
    try:
        with open(artifact_path, "r", encoding="utf-8") as f:
            return json.load(f).get("cache_key") == cache_key
    except (json.JSONDecodeError, OSError):
        return False


def file_base(audio_path: str) -> str:
    return os.path.splitext(os.path.basename(audio_path))[0]


def list_audio_files(audio_dir: str = AUDIO_DIR) -> List[str]:
    return sorted(os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav"))


def parse_stage_args(description: str, argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        help="Normalized 16 kHz mono WAV to process (repeatable). Defaults to every WAV in --audio-dir.",
    )
    parser.add_argument("--audio-dir", default=AUDIO_DIR)
    parser.add_argument("--force", action="store_true", help="Recompute even when a valid cache exists.")
    return parser.parse_args(argv)


def resolve_audio_files(args: argparse.Namespace) -> List[str]:
    files = args.files or list_audio_files(args.audio_dir)
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Audio file(s) not found: {missing}")
    return files


def load_env_file(env_path: str = ".env") -> None:
    """Loads key-value pairs from .env into os.environ without overriding existing values."""
    if not os.path.exists(env_path) and env_path == ".env":
        env_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")), ".env")
    if not os.path.exists(env_path):
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


def write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)
