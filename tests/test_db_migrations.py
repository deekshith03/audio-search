import os
import tempfile
import unittest
from unittest import mock

import psycopg2

from src.db import connection, migrate
from tests.db_support import ThrowawayDatabaseTestCase, requires_database


def write_files(directory, names):
    for name, body in names.items():
        with open(os.path.join(directory, name), "w", encoding="utf-8") as f:
            f.write(body)


class TestDatabaseUrl(unittest.TestCase):

    def test_uses_environment_when_set(self):
        with mock.patch.dict(os.environ, {"DATABASE_URL": "postgresql://u:p@db:5432/x"}):
            self.assertEqual(connection.database_url(), "postgresql://u:p@db:5432/x")

    def test_falls_back_to_local_compose_port(self):
        with mock.patch.dict(os.environ, {"DATABASE_URL": ""}):
            self.assertEqual(connection.database_url(), "postgresql://audio:audio@localhost:5433/audio_search")


class TestDiscoverMigrations(unittest.TestCase):

    def test_orders_by_version_and_ignores_non_sql_files(self):
        with tempfile.TemporaryDirectory() as d:
            write_files(d, {"002_b.sql": "", "001_a.sql": "", "README.md": ""})
            self.assertEqual([m.name for m in migrate.discover_migrations(d)], ["001_a.sql", "002_b.sql"])

    def test_rejects_badly_named_file(self):
        with tempfile.TemporaryDirectory() as d:
            write_files(d, {"1_init.sql": ""})
            with self.assertRaisesRegex(ValueError, "001_name.sql"):
                migrate.discover_migrations(d)

    def test_rejects_duplicate_versions(self):
        with tempfile.TemporaryDirectory() as d:
            write_files(d, {"001_a.sql": "", "001_b.sql": ""})
            with self.assertRaisesRegex(ValueError, "duplicate migration versions: 001"):
                migrate.discover_migrations(d)

    def test_repository_migrations_are_well_formed(self):
        versions = [m.version for m in migrate.discover_migrations()]
        self.assertEqual(versions[:3], ["001", "002", "003"])


@requires_database
class TestMigrationsAgainstDatabase(ThrowawayDatabaseTestCase):

    def setUp(self):
        self.conn = connection.connect(self.url)

    def tearDown(self):
        self.conn.close()

    def fetch_column(self, query, params=None):
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            return {row[0] for row in cur.fetchall()}

    def test_01_applies_all_migrations_then_nothing(self):
        applied = migrate.migrate(self.conn)
        self.assertEqual([m.version for m in applied], [m.version for m in migrate.discover_migrations()])
        self.assertEqual(migrate.migrate(self.conn), [])
        self.assertEqual(migrate.pending_migrations(self.conn), [])

    def test_02_extensions_installed(self):
        extensions = self.fetch_column("SELECT extname FROM pg_extension")
        self.assertTrue({"pg_search", "vector", "pg_trgm"} <= extensions)

    def test_03_schema_tables_and_indexes(self):
        tables = self.fetch_column("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        self.assertTrue({"files", "speakers", "sentences", "chunks", "chunk_embeddings", "schema_migrations"} <= tables)
        indexes = self.fetch_column("SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'")
        self.assertTrue({"chunks_bm25_idx", "chunks_text_trgm_idx", "chunks_file_start_idx"} <= indexes)

    def test_04_bm25_stems_and_filters_by_chunker(self):
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO files (workspace, file_id, sha256, pipeline_key) VALUES ('dataset', 'x.wav', 's', 'k') RETURNING id"
            )
            file_pk = cur.fetchone()[0]
            cur.executemany(
                "INSERT INTO chunks (file_pk, chunker, speaker_label, start_s, end_s, text, sentence_ids)"
                " VALUES (%s, %s, 'SPEAKER_00', 0, 1, %s, '{}')",
                [(file_pk, "A", "Wayland compositors are fast"), (file_pk, "B", "a compositor too"), (file_pk, "A", "unrelated")],
            )
            cur.execute(
                "SELECT text FROM chunks WHERE text ||| 'compositor' AND chunker = 'A' ORDER BY pdb.score(id) DESC"
            )
            self.assertEqual([r[0] for r in cur.fetchall()], ["Wayland compositors are fast"])
            cur.execute("SELECT text FROM chunks WHERE 'compositer' <% text ORDER BY text")
            self.assertEqual([r[0] for r in cur.fetchall()], ["a compositor too", "Wayland compositors are fast"])
        self.conn.rollback()

    def test_05_embeddings_accept_any_dimension(self):
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO files (workspace, file_id, sha256, pipeline_key) VALUES ('dataset', 'y.wav', 's', 'k') RETURNING id"
            )
            file_pk = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO chunks (file_pk, chunker, speaker_label, start_s, end_s, text, sentence_ids)"
                " VALUES (%s, 'D', 'SPEAKER_01', 0, 1, 'hi', '{}') RETURNING id",
                (file_pk,),
            )
            chunk_id = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO chunk_embeddings VALUES (%s, 'small', '[1,0,0]'), (%s, 'large', '[1,0,0,0,0]')",
                (chunk_id, chunk_id),
            )
            cur.execute("SELECT model, vector_dims(embedding) FROM chunk_embeddings ORDER BY model")
            self.assertEqual(cur.fetchall(), [("large", 5), ("small", 3)])
        self.conn.rollback()

    def test_06_deleting_a_file_cascades(self):
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO files (workspace, file_id, sha256, pipeline_key) VALUES ('data', 'z.wav', 's', 'k') RETURNING id"
            )
            file_pk = cur.fetchone()[0]
            cur.execute("INSERT INTO speakers VALUES (%s, 'SPEAKER_00', 'Ada')", (file_pk,))
            cur.execute(
                "INSERT INTO sentences (file_pk, speaker_label, turn_id, start_s, end_s, text) VALUES (%s, 'SPEAKER_00', 0, 0, 1, 'hi')",
                (file_pk,),
            )
            cur.execute("DELETE FROM files WHERE id = %s", (file_pk,))
            cur.execute("SELECT (SELECT count(*) FROM speakers) + (SELECT count(*) FROM sentences)")
            self.assertEqual(cur.fetchone()[0], 0)
        self.conn.rollback()

    def test_07_same_file_id_allowed_in_different_workspaces_only(self):
        with self.conn.cursor() as cur:
            insert = "INSERT INTO files (workspace, file_id, sha256, pipeline_key) VALUES (%s, 'dup.wav', 's', 'k')"
            cur.execute(insert, ("dataset",))
            cur.execute(insert, ("data",))
            with self.assertRaises(psycopg2.errors.UniqueViolation):
                cur.execute(insert, ("data",))
        self.conn.rollback()

    def test_08_failed_migration_is_rolled_back(self):
        with tempfile.TemporaryDirectory() as d:
            write_files(d, {"900_ok.sql": "CREATE TABLE ok_900 (id int);", "901_bad.sql": "CREATE TABLE bad_901 (id int); SELECT nope;"})
            with self.assertRaises(psycopg2.Error):
                migrate.migrate(self.conn, d)
            versions = self.fetch_column("SELECT version FROM schema_migrations")
            tables = self.fetch_column("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        self.assertIn("900", versions)
        self.assertNotIn("901", versions)
        self.assertIn("ok_900", tables)
        self.assertNotIn("bad_901", tables)


if __name__ == "__main__":
    unittest.main()
