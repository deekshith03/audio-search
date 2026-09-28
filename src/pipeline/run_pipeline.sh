#!/usr/bin/env bash
# Sequential Multi-Process Pipeline Runner on Apple Silicon
# Runs each stage as an isolated Python process to completely release Metal GPU / PyTorch memory between stages.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

# Ensure .venv is used
export PATH="${ROOT_DIR}/.venv/bin:$PATH"

# Load .env file if present
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

echo "=========================================================================="
echo "  STAGE 1A: Batch ASR Transcription on Apple Metal GPU (mlx-whisper)"
echo "=========================================================================="
python3 -m src.pipeline.asr

echo ""
echo "=========================================================================="
echo "  STAGE 1B: Word-Level Forced Alignment (wav2vec2 / WhisperX)"
echo "=========================================================================="
python3 -m src.pipeline.align

echo ""
echo "=========================================================================="
echo "  STAGE 2: Speaker Diarization via PyAnnote Community-1 (MPS / CPU)"
echo "=========================================================================="
python3 -m src.pipeline.diarize

echo ""
echo "=========================================================================="
echo "  STAGE 3: Deterministic Turn Reconciliation & Canonical Output Emission"
echo "=========================================================================="
python3 -m src.pipeline.reconcile

echo ""
echo "=========================================================================="
echo "  STAGE 4: Quality Evaluation (WER, DER 0ms/250ms & Word Speaker Accuracy)"
echo "=========================================================================="
python3 -m src.pipeline.evaluate_pipeline
