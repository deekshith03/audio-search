import json
import os
import shutil
import tempfile
import unittest

from src.db import connection
from src.pipeline.common import Workspace
from src.pipeline.labels import save_labels
from src.search.embedders import EmbeddingModel
from src.search.indexer import Indexer, canonical_paths, hnsw_index_name, vector_literal
from tests.db_support import ThrowawayDatabaseTestCase, requires_database

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_OUTPUTS = os.path.join(REPO_ROOT, "dataset", "pipeline_outputs")
SAMPLE_FILES = ("audio_01_lex_dhh_omarchy", "audio_05_sports_tactics_debate")


class FakeEmbedder:
    """Deterministic 3-d vectors; records every text it was asked to embed."""

    def __init__(self, key="fake", dimensions=3):
        self.model = EmbeddingModel(key, f"test/{key}", dimensions, 512)
        self.calls = []

    def encode_documents(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t) % 7 + 1)] + [1.0] * (self.model.dimensions - 1) for t in texts]


class TestHelpers(unittest.TestCase):

    def test_vector_literal(self):
        self.assertEqual(vector_literal([1.0, 0.5, -0.25]), "[1,0.5,-0.25]")

    def test_hnsw_index_name_is_a_safe_identifier(self):
        self.assertEqual(hnsw_index_name("gemma"), "chunk_embeddings_hnsw_gemma")
        self.assertEqual(hnsw_index_name("Some-Model"), "chunk_embeddings_hnsw_some_model")

    def test_canonical_paths_for_named_and_missing_files(self):
        golden = Workspace(os.path.join(REPO_ROOT, "dataset"))
        self.assertEqual(len(canonical_paths(golden)), 7)
        (path,) = canonical_paths(golden, ["dataset/audio/audio_01_lex_dhh_omarchy.wav"])
        self.assertTrue(path.endswith("audio_01_lex_dhh_omarchy_canonical.json"))
        with self.assertRaises(FileNotFoundError):
            canonical_paths(golden, ["missing.wav"])

    def test_canonical_paths_empty_workspace(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(canonical_paths(Workspace(d)), [])


@requires_database
class TestIndexer(ThrowawayDatabaseTestCase):
    apply_migrations = True

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.workspace = Workspace(self.tmp)
        os.makedirs(self.workspace.output_dir)
        for base in SAMPLE_FILES:
            shutil.copy(os.path.join(GOLDEN_OUTPUTS, f"{base}_canonical.json"), self.workspace.output_dir)
        self.conn = connection.connect(self.url)
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE files CASCADE")
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        shutil.rmtree(self.tmp)

    def indexer(self, *embedders):
        return Indexer(self.conn, self.workspace, embedders)

    def scalar(self, query, params=None):
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchone()[0]

    def paths(self):
        return canonical_paths(self.workspace)

    def test_indexes_files_sentences_and_a30s_chunks(self):
        stats = self.indexer().index_all(self.paths())
        self.assertTrue(all(s["rechunked"] for s in stats))
        self.assertEqual(self.scalar("SELECT count(*) FROM files WHERE workspace = %s", (self.tmp,)), 2)
        self.assertGreater(self.scalar("SELECT count(*) FROM sentences"), 100)
        with self.conn.cursor() as cur:
            cur.execute("SELECT DISTINCT chunker FROM chunks")
            self.assertEqual({r[0] for r in cur.fetchall()}, {"A-30s"})

    def test_sentence_word_timings_match_the_transcript(self):
        self.indexer().index_all(self.paths()[:1])
        with open(self.paths()[0], encoding="utf-8") as f:
            canonical = json.load(f)
        first_turn = canonical["turns"][0]
        with self.conn.cursor() as cur:
            cur.execute("SELECT text, start_s, end_s, words FROM sentences WHERE turn_id = %s ORDER BY start_s LIMIT 1",
                        (first_turn["turn_id"],))
            text, start_s, end_s, words = cur.fetchone()
            cur.execute("SELECT count(*) FROM sentences WHERE jsonb_array_length(words) = 0")
            self.assertEqual(cur.fetchone()[0], 0)
        self.assertEqual(" ".join(w[0] for w in words), text)
        self.assertEqual((words[0][1], words[-1][2]), (start_s, end_s))
        self.assertEqual(words[0], [first_turn["words"][0]["word"], first_turn["words"][0]["start_seconds"],
                                    first_turn["words"][0]["end_seconds"]])

    def test_chunk_sentence_ids_reference_sentences_of_the_same_file(self):
        self.indexer().index_all(self.paths())
        orphans = self.scalar(
            "SELECT count(*) FROM chunks c CROSS JOIN LATERAL unnest(c.sentence_ids) AS u(sid)"
            " LEFT JOIN sentences s ON s.id = u.sid AND s.file_pk = c.file_pk WHERE s.id IS NULL"
        )
        self.assertEqual(orphans, 0)

    def test_unchanged_files_are_skipped_and_force_rebuilds(self):
        self.indexer().index_all(self.paths())
        first_ids = self.scalar("SELECT array_agg(id ORDER BY id) FROM files")
        self.assertFalse(any(s["rechunked"] for s in self.indexer().index_all(self.paths())))
        self.assertEqual(self.scalar("SELECT array_agg(id ORDER BY id) FROM files"), first_ids)
        self.assertTrue(all(s["rechunked"] for s in self.indexer().index_all(self.paths(), force=True)))
        self.assertNotEqual(self.scalar("SELECT array_agg(id ORDER BY id) FROM files"), first_ids)

    def test_changed_transcript_is_rechunked(self):
        self.indexer().index_all(self.paths())
        path = self.paths()[0]
        with open(path, encoding="utf-8") as f:
            canonical = json.load(f)
        canonical["turns"] = canonical["turns"][:3]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(canonical, f)
        stats = self.indexer().index_all(self.paths())
        self.assertEqual([s["rechunked"] for s in stats], [True, False])
        self.assertEqual(self.scalar("SELECT count(DISTINCT c.speaker_label || s.turn_id) FROM chunks c"
                                     " JOIN sentences s ON s.id = c.sentence_ids[1]"
                                     " JOIN files f ON f.id = c.file_pk WHERE f.file_id = %s", (canonical["file_id"],)), 3)

    def test_embeds_every_chunk_and_sentence_text(self):
        embedder = FakeEmbedder()
        self.indexer(embedder).index_all(self.paths())
        self.assertEqual(self.scalar("SELECT count(*) FROM chunk_embeddings WHERE model = 'fake'"),
                         self.scalar("SELECT count(*) FROM chunks"))
        self.assertEqual(self.scalar("SELECT count(*) FROM sentence_embeddings WHERE model = 'fake'"),
                         self.scalar("SELECT count(*) FROM sentences"))
        embedded = {t for call in embedder.calls for t in call}
        with self.conn.cursor() as cur:
            cur.execute("SELECT text FROM chunks ORDER BY id LIMIT 5")
            self.assertTrue({r[0] for r in cur.fetchall()} <= embedded)

    def test_embeddings_are_incremental(self):
        self.indexer(FakeEmbedder()).index_all(self.paths())
        again = FakeEmbedder()
        stats = self.indexer(again).index_all(self.paths())
        self.assertEqual(again.calls, [])
        self.assertEqual(self.scalar("SELECT count(*) FROM sentence_embeddings"), self.scalar("SELECT count(*) FROM sentences"))
        self.assertTrue(all(s["embedded"]["fake"]["vectors"] == 0 for s in stats))
        other = FakeEmbedder("other", dimensions=5)
        self.indexer(other).index_all(self.paths())
        self.assertEqual(self.scalar("SELECT vector_dims(embedding) FROM chunk_embeddings WHERE model = 'other' LIMIT 1"), 5)

    def test_creates_partial_hnsw_index_per_model(self):
        self.indexer(FakeEmbedder()).index_all(self.paths()[:1])
        with self.conn.cursor() as cur:
            cur.execute("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'chunk_embeddings' AND indexname LIKE '%hnsw%'")
            indexes = dict(cur.fetchall())
        self.assertEqual(set(indexes), {"chunk_embeddings_hnsw_fake"})
        self.assertIn("vector(3)", indexes["chunk_embeddings_hnsw_fake"])
        self.assertIn("'fake'", indexes["chunk_embeddings_hnsw_fake"])

    def test_speaker_names_synced_from_labels_on_every_run(self):
        self.indexer().index_all(self.paths())
        file_id = "audio_01_lex_dhh_omarchy.wav"
        name_of = "SELECT s.display_name FROM speakers s JOIN files f ON f.id = s.file_pk WHERE f.file_id = %s AND speaker_label = 'SPEAKER_00'"
        self.assertIsNone(self.scalar(name_of, (file_id,)))
        save_labels(self.workspace, file_id, {"SPEAKER_00": "Lex", "SPEAKER_01": "DHH"}, ["SPEAKER_00", "SPEAKER_01"])
        stats = self.indexer().index_all(self.paths())
        self.assertFalse(any(s["rechunked"] for s in stats))
        self.assertEqual(self.scalar(name_of, (file_id,)), "Lex")

    def test_prune_removes_files_no_longer_on_disk(self):
        self.indexer().index_all(self.paths())
        os.remove(self.paths()[0])
        self.indexer().index_all(self.paths(), prune=True)
        self.assertEqual(self.scalar("SELECT count(*) FROM files"), 1)
        self.assertEqual(self.scalar("SELECT count(DISTINCT file_pk) FROM chunks"), 1)

    def test_workspaces_are_independent(self):
        self.indexer().index_all(self.paths())
        other_root = tempfile.mkdtemp()
        try:
            other = Workspace(other_root)
            os.makedirs(other.output_dir)
            shutil.copy(self.paths()[0], other.output_dir)
            Indexer(self.conn, other).index_all(canonical_paths(other), prune=True)
            self.assertEqual(self.scalar("SELECT count(*) FROM files"), 3)
        finally:
            shutil.rmtree(other_root)


if __name__ == "__main__":
    unittest.main()
