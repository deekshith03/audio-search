#!/usr/bin/env bash
# Reproduces the Phase 1-2 results end to end.
#
#   scripts/reproduce.sh              score the committed golden-set pipeline outputs (minutes)
#   scripts/reproduce.sh --recompute  re-run the full pipeline on all 7 files first (~40 min on CPU)
#
# Steps: audio provenance check -> dataset integrity -> [pipeline] -> pipeline quality scorecard -> unit tests

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
[ -x .venv/bin/python ] && PYTHON=.venv/bin/python

RECOMPUTE=0
for arg in "$@"; do
  case "$arg" in
    --recompute) RECOMPUTE=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 64 ;;
  esac
done

step() { echo; echo "=== $* ==="; }

step "1/5 Golden audio matches recorded provenance"
"$PYTHON" -m scripts.build_dataset --verify

step "2/5 Ground truth and qrels integrity"
"$PYTHON" evals/validate_dataset_integrity.py

if [ "$RECOMPUTE" = 1 ]; then
  step "3/5 Full pipeline re-run (all 7 files, --force)"
  "$PYTHON" -m scripts.bootstrap_models
  bash src/pipeline/run_pipeline.sh --force
else
  step "3/5 Pipeline quality scorecard on committed outputs (pass --recompute to regenerate)"
  "$PYTHON" -m src.pipeline.evaluate_pipeline
fi

step "4/5 Unit tests"
TEST_LOG="$(mktemp)"
if "$PYTHON" -m unittest discover >"$TEST_LOG" 2>&1; then TEST_STATUS=0; else TEST_STATUS=$?; fi
grep -E "^(Ran |OK|FAILED|FAIL:|ERROR:)" "$TEST_LOG" || true
if [ "$TEST_STATUS" != 0 ]; then echo "Unit tests failed; full log: $TEST_LOG" >&2; exit "$TEST_STATUS"; fi
rm -f "$TEST_LOG"

step "5/5 Done"
echo "Scorecard saved to dataset/pipeline_outputs/pipeline_manifest.json"
