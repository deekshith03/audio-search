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

if [ -f "${ROOT_DIR}/.env" ]; then
  set -a
  source "${ROOT_DIR}/.env"
  set +a
fi

TOKEN_FILE="${HF_TOKEN_PATH:-${HF_HOME:-$HOME/.cache/huggingface}/token}"
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_TOKEN:-}" ] && [ ! -f "$TOKEN_FILE" ]; then
  echo "ERROR: HF_TOKEN is not set in environment or .env, and no cached token found in ${TOKEN_FILE}."
  echo "Please set HF_TOKEN in .env or login using huggingface-cli."
  exit 1
fi

run_stage() {
  local title="$1"; shift
  echo ""
  echo "=========================================================================="
  echo "  ${title}"
  echo "=========================================================================="
  uv run python -m "$@"
}

run_stage "STAGE 1A: ASR (faster-whisper large-v3-turbo, fp32, CPU)" src.pipeline.asr "$@"
run_stage "STAGE 1B: Word-Level Forced Alignment (wav2vec2, CPU)" src.pipeline.align "$@"
run_stage "STAGE 2: Speaker Diarization (pyannote community-1, k=2, CPU)" src.pipeline.diarize "$@"
run_stage "STAGE 3: Word-to-Speaker Turn Reconciliation" src.pipeline.reconcile "$@"
run_stage "STAGE 4: Quality Evaluation (WER, DER raw/refined/reconciled, Word Speaker Accuracy)" src.pipeline.evaluate_pipeline "$@"
