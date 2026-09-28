#!/usr/bin/env bash
# Sequential pipeline runner. Each stage runs as its own Python process with its own cache,
# so a failed or changed stage can be re-run in isolation.
#
# Usage: src/pipeline/run_pipeline.sh [--file dataset/audio/x.wav ...] [--force]
# Arguments are forwarded to every stage. Stage 4 (evaluation) runs only for files with ground truth.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
  PYTHON="${ROOT_DIR}/.venv/bin/python"
else
  PYTHON="python3"
fi

if [ -f "${ROOT_DIR}/.env" ]; then
  set -a
  source "${ROOT_DIR}/.env"
  set +a
fi

if [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HOME/.cache/huggingface/token" ]; then
  echo "ERROR: HF_TOKEN is not set in environment or .env, and no cached token found in ~/.cache/huggingface/token."
  echo "Please set HF_TOKEN in .env or login using huggingface-cli."
  exit 1
fi

run_stage() {
  local title="$1"; shift
  echo ""
  echo "=========================================================================="
  echo "  ${title}"
  echo "=========================================================================="
  "${PYTHON}" -m "$@"
}

run_stage "STAGE 1A: ASR (faster-whisper large-v3-turbo, fp32, CPU)" src.pipeline.asr "$@"
run_stage "STAGE 1B: Word-Level Forced Alignment (wav2vec2, CPU)" src.pipeline.align "$@"
run_stage "STAGE 2: Speaker Diarization (pyannote community-1, k=2, CPU)" src.pipeline.diarize "$@"
run_stage "STAGE 3: Word-to-Speaker Turn Reconciliation" src.pipeline.reconcile "$@"
run_stage "STAGE 4: Quality Evaluation (WER, DER raw/refined/reconciled, Word Speaker Accuracy)" src.pipeline.evaluate_pipeline "$@"
