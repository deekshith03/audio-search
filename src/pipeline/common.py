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
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Workspace:
    """Directory layout shared by the golden set (`dataset/`) and user uploads (`data/`). The root is
    normalized ("dataset/" and "./dataset" are "dataset") because it is also the workspace key
    stored with every indexed file."""

    root: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", os.path.normpath(self.root))

    @property
    def audio_dir(self) -> str:
        return os.path.join(self.root, "audio")

    @property
    def raw_asr_dir(self) -> str:
        return os.path.join(self.root, "cache", "raw_asr")

    @property
    def diarization_dir(self) -> str:
        return os.path.join(self.root, "cache", "raw_diarization")

    @property
    def output_dir(self) -> str:
        return os.path.join(self.root, "pipeline_outputs")

    @property
    def labels_dir(self) -> str:
        return os.path.join(self.root, "speaker_labels")

    @property
    def jobs_dir(self) -> str:
        return os.path.join(self.root, "jobs")

    @property
    def uploads_dir(self) -> str:
        return os.path.join(self.root, "uploads")

    def canonical_path(self, file_base_name: str) -> str:
        return os.path.join(self.output_dir, f"{file_base_name}_canonical.json")


GOLDEN = Workspace("dataset")
AUDIO_DIR = GOLDEN.audio_dir
RAW_ASR_DIR = GOLDEN.raw_asr_dir
DIARIZATION_DIR = GOLDEN.diarization_dir
OUTPUT_DIR = GOLDEN.output_dir
GROUND_TRUTH_DIR = os.path.join(GOLDEN.root, "ground_truth")


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
        help="Normalized 16 kHz mono WAV to process (repeatable). Defaults to every WAV in the workspace audio dir.",
    )
    parser.add_argument("--workspace", default=GOLDEN.root, help="Workspace root (dataset/ for the golden set, data/ for uploads).")
    parser.add_argument("--audio-dir", default=None, help="Overrides <workspace>/audio.")
    parser.add_argument("--force", action="store_true", help="Recompute even when a valid cache exists.")
    args = parser.parse_args(argv)
    args.workspace = Workspace(args.workspace)
    args.audio_dir = args.audio_dir or args.workspace.audio_dir
    return args


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


def hf_token_path() -> str:
    """Where `huggingface-cli login` saves the token: $HF_TOKEN_PATH, else $HF_HOME/token (the
    Docker image sets HF_HOME=/models/huggingface), else ~/.cache/huggingface/token."""
    if os.environ.get("HF_TOKEN_PATH"):
        return os.environ["HF_TOKEN_PATH"]
    return os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "token")


def get_hf_token(env_path: Optional[str] = ".env") -> str:
    if env_path:
        load_env_file(env_path)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        token_path = hf_token_path()
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
    """Atomic write through a uniquely named temp file, so concurrent writers never share one."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=f".{os.path.basename(path)}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
