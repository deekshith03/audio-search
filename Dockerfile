# syntax=docker/dockerfile:1.7
# Audio Search: CPU-only image for the transcription pipeline and Streamlit app.
# Builds natively for linux/arm64 (Apple Silicon) and linux/amd64.

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_NO_SYNC=1 \
    PATH="/opt/venv/bin:${PATH}" \
    MODEL_CACHE_DIR=/models \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    APP_DATA_DIR=/app/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.4 /uv /usr/local/bin/uv

# The app runs as this user; the entrypoint starts as root only to fix volume ownership, then drops to it.
RUN useradd --create-home --uid 1000 app

WORKDIR /app

# Dependencies first so code changes do not invalidate the (large) dependency layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY --chown=app:app . .

RUN chmod +x scripts/docker-entrypoint.sh scripts/reproduce.sh src/pipeline/run_pipeline.sh \
    && mkdir -p /models /app/data \
    && chown app:app /models /app/data

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
    CMD uv run python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

ENTRYPOINT ["scripts/docker-entrypoint.sh"]
CMD ["app"]
