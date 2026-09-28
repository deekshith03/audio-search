"""
Downloads every model the pipeline needs into the model cache (HF_HOME / TORCH_HOME), once.

  python -m scripts.bootstrap_models

Exit codes: 0 ready, 2 no Hugging Face token, 3 token invalid, 4 pyannote terms not accepted,
1 any other download failure. A marker file records a successful bootstrap so later container
starts skip straight to the app (and work offline).
"""

import json
import os
import sys
from typing import Callable, List, Tuple

from src.pipeline.align import ALIGN_MODEL
from src.pipeline.asr import MODEL_NAME as ASR_MODEL_NAME
from src.pipeline.asr import MODEL_REPO as ASR_MODEL_REPO
from src.pipeline.common import load_env_file
from src.pipeline.diarize import DIARIZATION_MODEL

PYANNOTE_TERMS_URL = f"https://huggingface.co/{DIARIZATION_MODEL}"
TOKEN_SETTINGS_URL = "https://huggingface.co/settings/tokens"


def marker_path() -> str:
    root = os.environ.get("MODEL_CACHE_DIR") or os.path.dirname(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    return os.path.join(root, ".bootstrap.json")


def expected_marker() -> dict:
    return {"asr": ASR_MODEL_REPO, "alignment": ALIGN_MODEL, "diarization": DIARIZATION_MODEL}


def already_bootstrapped() -> bool:
    try:
        with open(marker_path(), "r", encoding="utf-8") as f:
            return json.load(f) == expected_marker()
    except (OSError, json.JSONDecodeError):
        return False


def fail(code: int, message: str) -> None:
    print(f"\n✗ {message}\n", file=sys.stderr)
    sys.exit(code)


def download_diarization(token: str) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    try:
        snapshot_download(DIARIZATION_MODEL, token=token)
    except GatedRepoError:
        fail(4, f"Your Hugging Face account has not accepted the terms for {DIARIZATION_MODEL}.\n"
                f"  Open {PYANNOTE_TERMS_URL}, log in, accept the conditions, then restart.")
    except RepositoryNotFoundError:
        fail(3, f"Hugging Face could not authorize access to {DIARIZATION_MODEL}. Check HF_TOKEN ({TOKEN_SETTINGS_URL}).")
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            fail(3, f"HF_TOKEN was rejected by Hugging Face. Create a read token at {TOKEN_SETTINGS_URL}.")
        raise


def download_asr() -> None:
    from faster_whisper.utils import download_model

    download_model(ASR_MODEL_NAME)


def download_alignment() -> None:
    import torchaudio

    getattr(torchaudio.pipelines, ALIGN_MODEL).get_model()


def main() -> None:
    if already_bootstrapped():
        print("✓ Models already downloaded.")
        return

    load_env_file()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        fail(2, "HF_TOKEN is not set. Speaker diarization uses a gated pyannote model:\n"
                f"  1. Create a read token at {TOKEN_SETTINGS_URL}\n"
                f"  2. Accept the terms at {PYANNOTE_TERMS_URL}\n"
                "  3. Put HF_TOKEN=hf_... in .env and restart.")

    steps: List[Tuple[str, Callable[[], None]]] = [
        (f"Speaker diarization  {DIARIZATION_MODEL}", lambda: download_diarization(token)),
        (f"Speech recognition   {ASR_MODEL_REPO} (~1.6 GB)", download_asr),
        (f"Word alignment       torchaudio {ALIGN_MODEL} (~360 MB)", download_alignment),
    ]
    print("Downloading models (first start only)...")
    for i, (label, step) in enumerate(steps, start=1):
        print(f"  [{i}/{len(steps)}] {label}", flush=True)
        try:
            step()
        except SystemExit:
            raise
        except Exception as e:
            fail(1, f"Download failed for {label.strip()}: {e}")

    os.makedirs(os.path.dirname(marker_path()), exist_ok=True)
    with open(marker_path(), "w", encoding="utf-8") as f:
        json.dump(expected_marker(), f)
    print("✓ All models ready.")


if __name__ == "__main__":
    main()
