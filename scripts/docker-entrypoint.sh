#!/usr/bin/env bash
# Container entrypoint.
#
#   app                 apply DB migrations, download models on first start, index the golden set
#                       (incremental; seconds when nothing changed), then serve Streamlit on :8501 (default)
#   reproduce [--recompute]   provenance + integrity checks, scorecard, unit tests
#   pipeline [args]     src/pipeline/run_pipeline.sh (e.g. --file dataset/audio/x.wav --force)
#   test                unit tests
#   bootstrap           download models only
#   migrate [status]    apply (or list) database schema migrations
#   anything else       executed as-is (e.g. bash)

set -euo pipefail
cd /app

command="${1:-app}"
[ "$#" -gt 0 ] && shift

case "$command" in
  app)
    uv run python -m src.db.migrate
    uv run python -m scripts.bootstrap_models
    uv run python -m src.search.indexer --workspace dataset
    exec uv run streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501 "$@"
    ;;
  reproduce)
    exec scripts/reproduce.sh "$@"
    ;;
  pipeline)
    uv run python -m scripts.bootstrap_models
    exec bash src/pipeline/run_pipeline.sh "$@"
    ;;
  test)
    exec uv run python -m unittest discover "$@"
    ;;
  bootstrap)
    exec uv run python -m scripts.bootstrap_models
    ;;
  migrate)
    exec uv run python -m src.db.migrate "$@"
    ;;
  *)
    exec "$command" "$@"
    ;;
esac
