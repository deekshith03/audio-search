"""
PostgreSQL connection for the search index (ParadeDB: pg_search + pgvector + pg_trgm).

Docker sets DATABASE_URL to the `db` service; native runs fall back to the compose port on
localhost, so `docker compose up -d db` plus `uv run ...` works without extra configuration.
"""

import os
from typing import Optional

import psycopg2

DEFAULT_DATABASE_URL = "postgresql://audio:audio@localhost:5433/audio_search"


def database_url() -> str:
    return os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL


def connect(url: Optional[str] = None, connect_timeout: int = 5):
    return psycopg2.connect(url or database_url(), connect_timeout=connect_timeout)
