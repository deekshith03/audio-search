import json
import math
import os
import re
import shutil
import tempfile
import unittest
import zlib
from unittest import mock

from evals import evaluate_recall, search_provider
from src.db import connection
from src.pipeline.common import Workspace
from src.pipeline.labels import save_labels
from src.search.embedders import EMBEDDING_MODEL
from src.search.engine import SearchConfig, SearchEngine
from src.search.indexer import Indexer, canonical_paths
from tests.db_support import ThrowawayDatabaseTestCase, requires_database


class HashEmbedder:
    """Bag-of-words vectors in EmbeddingGemma's 768 dimensions: texts sharing words are close."""

    model = EMBEDDING_MODEL

    def __init__(self):
        self.queries = []

    def vector(self, text):
        v = [0.0] * self.model.dimensions
        for word in re.findall(r"\w+", text.lower()):
            v[zlib.crc32(word.encode()) % self.model.dimensions] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def encode_documents(self, texts):
        return [self.vector(t) for t in texts]

    def encode_query(self, text):
        self.queries.append(text)
        return self.vector(text)


class NoQueryEmbedder(HashEmbedder):
    def encode_query(self, text):
        raise AssertionError("lexical mode must not embed the query")


def turn(turn_id, speaker, text, start):
    words, t = [], start
    for token in text.split():
        words.append({"word": token, "start_seconds": round(t, 2), "end_seconds": round(t + 0.35, 2)})
        t += 0.4
    return {"turn_id": turn_id, "speaker_label": speaker, "start_seconds": start, "end_seconds": t, "words": words}


def canonical(file_id, turns):
    return {"file_id": file_id, "audio_duration_seconds": 600.0, "speaker_labels": ["SPEAKER_00", "SPEAKER_01"], "turns": turns}


LINUX = canonical("linux_talk.wav", [
    turn(1, "SPEAKER_00", "So what desktop are you running these days on your laptop at home?", 0.0),
    turn(2, "SPEAKER_01", "I switched to Hyperland which is a tiling Wayland compositor. "
                          "It makes every window snap into place without a mouse. "
                          "The configuration lives in one plain text file that I edit by hand. "
                          "Honestly it changed how I think about my whole workflow.", 6.0),
    turn(3, "SPEAKER_00", "That sounds great. Did you try the zebra theme that everyone talks about?", 60.0),
    turn(4, "SPEAKER_01", "I did try the zebra theme once. It was far too loud for me so I went back to plain dark colors.", 70.0),
])
COOKING = canonical("cooking_show.wav", [
    turn(1, "SPEAKER_00", "What is the secret to a good sourdough bread at home?", 0.0),
    turn(2, "SPEAKER_01", "Patience is the secret. The starter needs days to mature and the dough needs a long cold proof. "
                          "Most people rush the fermentation and then wonder why the crumb is dense.", 5.0),
])


@requires_database
class TestSearchEngine(ThrowawayDatabaseTestCase):
    apply_migrations = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmp = tempfile.mkdtemp()
        cls.workspace = Workspace(cls.tmp)
        os.makedirs(cls.workspace.output_dir)
        for doc in (LINUX, COOKING):
            base = doc["file_id"].removesuffix(".wav")
            with open(cls.workspace.canonical_path(base), "w", encoding="utf-8") as f:
                json.dump(doc, f)
        save_labels(cls.workspace, "linux_talk.wav", {"SPEAKER_00": "Lex", "SPEAKER_01": "DHH"}, ["SPEAKER_00", "SPEAKER_01"])
        cls.conn = connection.connect(cls.url)
        Indexer(cls.conn, cls.workspace, [HashEmbedder()]).index_all(canonical_paths(cls.workspace))

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()
        shutil.rmtree(cls.tmp)
        super().tearDownClass()

    def engine(self, embedder=None):
        return SearchEngine(self.conn, embedder=embedder or HashEmbedder())

    def run_search(self, query, mode="hybrid", top_k=5, engine=None, **config):
        return (engine or self.engine()).search(query, mode, top_k, SearchConfig(**config), workspaces=(self.tmp,))

    def test_result_contract(self):
        (top, *_) = self.run_search("tiling window compositor").results
        self.assertEqual(top["file_id"], "linux_talk.wav")
        for key in ("file_id", "speaker", "speaker_label", "start_seconds", "end_seconds", "text", "highlight", "score", "rank"):
            self.assertIn(key, top)
        self.assertEqual(top["rank"], 1)
        self.assertLess(top["start_seconds"], top["end_seconds"])

    def test_speaker_name_joined_or_label_fallback(self):
        linux = self.run_search("tiling window compositor").results[0]
        self.assertEqual((linux["speaker"], linux["speaker_label"]), ("DHH", "SPEAKER_01"))
        cooking = self.run_search("sourdough fermentation crumb").results[0]
        self.assertEqual(cooking["speaker"], "SPEAKER_01")

    def test_every_mode_returns_results(self):
        for mode in ("hybrid", "lexical", "dense"):
            results = self.run_search("sourdough starter fermentation", mode=mode).results
            self.assertTrue(results, mode)
            self.assertEqual(results[0]["file_id"], "cooking_show.wav", mode)

    def test_lexical_mode_never_embeds_the_query(self):
        results = self.run_search("sourdough", mode="lexical", engine=self.engine(NoQueryEmbedder())).results
        self.assertEqual(results[0]["file_id"], "cooking_show.wav")

    def test_trigram_finds_asr_misspelling_in_short_queries(self):
        results = self.run_search("Hyprland", mode="lexical", bm25_weight=0.0).results
        self.assertEqual(results[0]["file_id"], "linux_talk.wav")
        self.assertIn("Hyperland", results[0]["text"])

    def test_trigram_skipped_for_long_queries(self):
        results = self.run_search("Hyprland desktop setup on the laptop today", mode="lexical", bm25_weight=0.0).results
        self.assertEqual(results, [])

    def test_results_are_pinpointed_and_near_their_chunk(self):
        for r in self.run_search("configuration plain text file edit by hand").results:
            self.assertLessEqual(r["end_seconds"] - r["start_seconds"], 20.0 + 1e-6)
            self.assertGreaterEqual(r["start_seconds"], r["chunk_start_seconds"] - 20.0)

    def test_no_overlapping_results_within_a_file(self):
        results = self.run_search("theme zebra plain dark colors window", top_k=10).results
        linux = sorted((r["start_seconds"], r["end_seconds"]) for r in results if r["file_id"] == "linux_talk.wav")
        for (_, end), (start, _) in zip(linux, linux[1:]):
            self.assertLessEqual(end, start)

    def test_highlight_marks_stemmed_matches_and_escapes_html(self):
        top = self.run_search("compositors", mode="lexical").results[0]
        self.assertIn("<mark>compositor.</mark>".replace(".", ""), top["highlight"].replace(".", ""))
        self.assertNotIn("<script>", top["highlight"])

    def test_short_keyword_query_is_tightened_to_the_matched_word(self):
        top = self.run_search("Hyprland", mode="lexical").results[0]
        self.assertIn("Hyperland", top["text"])
        self.assertLessEqual(top["end_seconds"] - top["start_seconds"], 6.0)
        untightened = self.run_search("Hyprland", mode="lexical", keyword_span_padding_seconds=0).results[0]
        self.assertGreater(untightened["end_seconds"] - untightened["start_seconds"], top["end_seconds"] - top["start_seconds"])

    def test_long_queries_and_dense_mode_are_not_tightened(self):
        for mode, query in (("hybrid", "tiling window compositor for the desktop"), ("dense", "Hyprland")):
            loose = self.run_search(query, mode=mode, keyword_span_padding_seconds=0).results
            tight = self.run_search(query, mode=mode).results
            self.assertEqual([(r["start_seconds"], r["end_seconds"]) for r in loose],
                             [(r["start_seconds"], r["end_seconds"]) for r in tight], mode)

    def test_workspace_filter(self):
        response = self.engine().search("sourdough", "hybrid", 5, SearchConfig(), workspaces=("somewhere-else",))
        self.assertEqual(response.results, [])

    def test_blank_query_returns_nothing(self):
        self.assertEqual(self.run_search("  ?!  ").results, [])

    def test_unknown_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown mode"):
            self.run_search("x", mode="fuzzy")

    def test_timings_and_fused_ids_reported(self):
        response = self.run_search("sourdough bread")
        self.assertTrue({"embed_query", "retrieve", "fuse", "localize", "total"} <= set(response.timings_ms))
        self.assertTrue(response.fused_chunk_ids)

    def test_zero_weight_retriever_is_skipped(self):
        self.assertEqual(self.run_search("Hyprland desktop setup on the laptop today", mode="lexical", bm25_weight=0.0).results, [])


class TestSearchConfig(unittest.TestCase):

    def test_frozen_defaults(self):
        config = SearchConfig()
        self.assertEqual(config.weights, {"bm25": 1.0, "trigram": 1.0, "dense": 2.0})
        self.assertEqual((config.candidates, config.localize_depth, config.span_extend_ratio, config.span_min_seconds),
                         (50, 20, 0.8, 4.0))

    def test_from_dict(self):
        self.assertEqual(SearchConfig.from_dict({"dense_weight": 1.5}).weights["dense"], 1.5)
        with self.assertRaisesRegex(ValueError, "unknown search config key"):
            SearchConfig.from_dict({"reranker": "bge-reranker"})


class TestSearchProvider(unittest.TestCase):

    def test_mock_mode_returns_empty_baseline(self):
        with mock.patch.dict(os.environ, {"EVAL_MOCK_MODE": "1"}):
            out = search_provider.call_api("q", {"config": {"mode": "dense", "top_k": 3}}, {})
        self.assertEqual(out["output"]["results"], [])
        self.assertEqual(out["output"]["status"], "mock_baseline")

    def test_passes_mode_top_k_and_config_overrides(self):
        with mock.patch.dict(os.environ, {"EVAL_MOCK_MODE": ""}), \
                mock.patch("src.search.engine.search", return_value=[{"file_id": "x.wav"}]) as search:
            out = search_provider.call_api(" q ", {"config": {"mode": "lexical", "top_k": 7, "dense_weight": 1.5, "basePath": "/repo"}}, {})
        kwargs = search.call_args.kwargs
        self.assertEqual((kwargs["query"], kwargs["mode"], kwargs["top_k"]), ("q", "lexical", 7))
        self.assertEqual(kwargs["config"].dense_weight, 1.5)
        self.assertEqual(kwargs["workspaces"], ("dataset",))
        self.assertEqual(out["output"]["results"], [{"file_id": "x.wav"}])

    def test_engine_failure_is_an_error_not_empty_results(self):
        with mock.patch.dict(os.environ, {"EVAL_MOCK_MODE": ""}), \
                mock.patch("src.search.engine.search", side_effect=RuntimeError("db down")):
            out = search_provider.call_api("q", {"config": {}}, {})
        self.assertNotIn("output", out)
        self.assertIn("db down", out["error"])

    def test_evaluate_recall_fails_closed_on_provider_error(self):
        with mock.patch.object(evaluate_recall, "call_api", return_value={"error": "db down"}):
            with self.assertRaisesRegex(RuntimeError, "db down"):
                evaluate_recall.run_benchmark(split="dev", modes=["hybrid"])

    def test_evaluate_recall_forwards_search_config(self):
        with mock.patch.object(evaluate_recall, "call_api", return_value={"output": {"results": []}}) as call:
            evaluate_recall.run_benchmark(split="dev", modes=["dense"], search_config={"dense_weight": 1.5, "mode": "ignored"})
        self.assertEqual(call.call_args.kwargs["options"]["config"], {"dense_weight": 1.5, "mode": "dense", "top_k": 5})

    def test_bad_config_is_an_error(self):
        with mock.patch.dict(os.environ, {"EVAL_MOCK_MODE": ""}):
            out = search_provider.call_api("q", {"config": {"chunkr": "B"}}, {})
        self.assertIn("unknown search config key", out["error"])


if __name__ == "__main__":
    unittest.main()
