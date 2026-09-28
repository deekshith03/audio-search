"""
Minimal schema migrations: numbered SQL files in db/migrations/, applied in order, once.

    uv run python -m src.db.migrate            # apply pending migrations (same as `up`)
    uv run python -m src.db.migrate status     # list applied and pending versions

Each file runs in its own transaction together with its `schema_migrations` row, so a failing
migration leaves nothing half-applied. Applied files are never edited; changes go in a new file.
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Set

from src.db.connection import connect, database_url

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIGRATIONS_DIR = os.path.join(REPO_ROOT, "db", "migrations")
MIGRATION_FILE_PATTERN = re.compile(r"^(\d{3})_[a-z0-9_]+\.sql$")

CREATE_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


@dataclass(frozen=True)
class Migration:
    version: str
    name: str
    path: str


def discover_migrations(directory: str = MIGRATIONS_DIR) -> List[Migration]:
    migrations = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".sql"):
            continue
        match = MIGRATION_FILE_PATTERN.match(filename)
        if not match:
            raise ValueError(f"migration file name must look like 001_name.sql: {filename}")
        migrations.append(Migration(match.group(1), filename, os.path.join(directory, filename)))
    versions = [m.version for m in migrations]
    duplicates = sorted({v for v in versions if versions.count(v) > 1})
    if duplicates:
        raise ValueError(f"duplicate migration versions: {', '.join(duplicates)}")
    return migrations


def applied_versions(conn) -> Set[str]:
    with conn.cursor() as cur:
        cur.execute(CREATE_MIGRATIONS_TABLE)
        cur.execute("SELECT version FROM schema_migrations")
        versions = {row[0] for row in cur.fetchall()}
    conn.commit()
    return versions


def pending_migrations(conn, directory: str = MIGRATIONS_DIR) -> List[Migration]:
    done = applied_versions(conn)
    return [m for m in discover_migrations(directory) if m.version not in done]


def migrate(conn, directory: str = MIGRATIONS_DIR) -> List[Migration]:
    applied = []
    for migration in pending_migrations(conn, directory):
        with open(migration.path, encoding="utf-8") as f:
            sql = f.read()
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (version, name) VALUES (%s, %s)",
                    (migration.version, migration.name),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        applied.append(migration)
    return applied


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Apply database schema migrations.")
    parser.add_argument("command", nargs="?", default="up", choices=["up", "status"])
    args = parser.parse_args(argv)

    conn = connect()
    try:
        if args.command == "status":
            done = applied_versions(conn)
            for m in discover_migrations():
                print(f"{'applied' if m.version in done else 'pending'}  {m.name}")
            return 0
        applied = migrate(conn)
    finally:
        conn.close()

    for m in applied:
        print(f"applied  {m.name}")
    if not applied:
        print(f"schema up to date ({database_url().rsplit('@', 1)[-1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
