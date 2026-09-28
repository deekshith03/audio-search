"""Throwaway PostgreSQL databases for tests, so the real search index is never touched."""

import unittest
import uuid

import psycopg2
from psycopg2 import sql

from src.db import connection, migrate


def server_reachable() -> bool:
    try:
        connection.connect(connect_timeout=2).close()
        return True
    except psycopg2.OperationalError:
        return False


requires_database = unittest.skipUnless(server_reachable(), "PostgreSQL not reachable (docker compose up -d db)")


def _admin(statement: sql.Composable) -> None:
    admin = connection.connect()
    admin.autocommit = True
    try:
        with admin.cursor() as cur:
            cur.execute(statement)
    finally:
        admin.close()


class ThrowawayDatabaseTestCase(unittest.TestCase):
    """Creates an empty database per test class; `apply_migrations` loads the real schema."""

    apply_migrations = False

    @classmethod
    def setUpClass(cls):
        cls.db_name = f"audio_search_test_{uuid.uuid4().hex[:8]}"
        _admin(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(cls.db_name)))
        cls.url = f"{connection.database_url().rsplit('/', 1)[0]}/{cls.db_name}"
        if cls.apply_migrations:
            conn = connection.connect(cls.url)
            migrate.migrate(conn)
            conn.close()

    @classmethod
    def tearDownClass(cls):
        _admin(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(cls.db_name)))
