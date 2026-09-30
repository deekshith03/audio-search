#!/usr/bin/env bash
# Runner for Conversational Audio Hybrid Search Evaluation Benchmark
#
#   bash evals/run_evals.sh [--split dev] [--mock] [--enforce-gate] [promptfoo eval args...]
#   bash evals/run_evals.sh --view      open the promptfoo results UI
#
# Only dev (the tuning split) runs here. The blind split is one-time: its result is in
# evals/results/final_blind.json and evals/evaluate_recall.py refuses a rerun.
#
# Promptfoo is a Node tool run through a pinned npx (no package.json / node_modules);
# Node itself is pinned by .mise.toml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PROMPTFOO_VERSION="0.123.1"
SPLIT="dev"
PROMPTFOO=(npx --registry https://registry.npmjs.org --yes "promptfoo@${PROMPTFOO_VERSION}")

# Dynamically resolve Node >= 22
if command -v node >/dev/null 2>&1 && [ "$(node -v | cut -d. -f1 | tr -d 'v')" -ge 22 ]; then
    NODE_BIN="$(dirname "$(command -v node)")"
elif [ -d "${HOME}/.local/share/mise/installs/node/22.23.0/bin" ]; then
    NODE_BIN="${HOME}/.local/share/mise/installs/node/22.23.0/bin"
else
    NODE_BIN="$(dirname "$(command -v node || echo '/usr/local/bin')")"
fi
export PATH="${NODE_BIN}:$PATH"

if [ "${1:-}" == "--view" ]; then
    exec "${PROMPTFOO[@]}" view
fi

PROMPTFOO_ARGS=()
EVAL_ARGS=()
MOCK_MODE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --mock) MOCK_MODE=1 ;;
        --enforce-gate) EVAL_ARGS+=("--enforce-gate") ;;
        --split=*) SPLIT="${1#--split=}" ;;
        --split)
            if [ $# -lt 2 ]; then echo "--split needs a value (dev)" >&2; exit 64; fi
            shift; SPLIT="$1" ;;
        *) PROMPTFOO_ARGS+=("$1") ;;
    esac
    shift
done

case "$SPLIT" in
    dev) ;;
    blind) echo "The blind split is one-time and already run (evals/results/final_blind.json); it is not rerun here." >&2; exit 64 ;;
    *) echo "Unknown --split '${SPLIT}' (expected dev)" >&2; exit 64 ;;
esac
EVAL_ARGS+=("--split" "$SPLIT")
PROMPTFOO_CONFIG="promptfooconfig.${SPLIT}.yaml"

echo "================================================================="
echo "  STAGE 1: Validating Dataset & Ground-Truth Integrity"
echo "================================================================="
uv run python evals/validate_dataset_integrity.py

echo ""
echo "================================================================="
echo "  STAGE 2: Promptfoo test matrix [split: ${SPLIT}] (queries x hybrid/lexical/dense)"
echo "================================================================="


if [ $MOCK_MODE -eq 1 ]; then
    export EVAL_MOCK_MODE=1
fi


# Promptfoo launches the Python provider itself; point it at the project environment.
export PROMPTFOO_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')"

promptfoo_exit=0
if [ ${#PROMPTFOO_ARGS[@]} -gt 0 ]; then
    "${PROMPTFOO[@]}" eval --no-cache -c "$PROMPTFOO_CONFIG" "${PROMPTFOO_ARGS[@]}" || promptfoo_exit=$?
else
    "${PROMPTFOO[@]}" eval --no-cache -c "$PROMPTFOO_CONFIG" || promptfoo_exit=$?
fi

echo ""
echo "================================================================="
echo "  STAGE 3: Computing Mathematical Recall@k & MRR Scorecard"
echo "================================================================="
eval_exit=0
uv run python evals/evaluate_recall.py "${EVAL_ARGS[@]}" || eval_exit=$?

# Fail-closed enforcement: exit with failure if either stage failed (except expected assertion failures during mock baseline)
if [ $MOCK_MODE -ne 1 ] && [ $promptfoo_exit -ne 0 ]; then
    echo "❌ Promptfoo test matrix failed with exit code $promptfoo_exit"
    exit $promptfoo_exit
fi

if [ $eval_exit -ne 0 ]; then
    echo "❌ Mathematical evaluation / gate check failed with exit code $eval_exit"
    exit $eval_exit
fi
