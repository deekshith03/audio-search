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
from src.search.embedders import MODELS
from src.search.engine import SearchConfig, SearchEngine
from src.search.indexer import Indexer, canonical_paths
from tests.db_support import ThrowawayDatabaseTestCase, requires_database


class HashEmbedder:
    """Bag-of-words vectors in bge-small's 384 dimensions: texts sharing words are close."""

    model = MODELS["bge-small"]

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


class SpyReranker:
    def __init__(self):
        self.calls = []

    def score(self, query, passages):
        self.calls.append(list(passages))
        return [0.0] * len(passages)


class KeywordReranker:
    """Scores a passage by whether it contains a marker word, to prove reranking reorders."""

    def __init__(self, marker):
        self.marker = marker

    def score(self, query, passages):
        return [1.0 if self.marker in p.lower() else 0.0 for p in passages]


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


def one_token_per_word(word):
    return 1


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
        Indexer(cls.conn, cls.workspace, [HashEmbedder()], count_tokens=one_token_per_word).index_all(canonical_paths(cls.workspace))

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()
        shutil.rmtree(cls.tmp)
        super().tearDownClass()

    def engine(self, embedder=None, rerankers=None):
        return SearchEngine(self.conn, embedders={"bge-small": embedder or HashEmbedder()}, rerankers=rerankers)

    # The fixture index is built with a fake bge-small embedder; pin the pre-freeze knobs these tests exercise.
    BASE = {"model": "bge-small", "context": True, "fusion": "rrf", "dense_weight": 1.0, "dedupe_gap_seconds": 2.0}

    def run_search(self, query, mode="hybrid", top_k=5, engine=None, **config):
        return (engine or self.engine()).search(query, mode, top_k, SearchConfig(**{**self.BASE, **config}), workspaces=(self.tmp,))

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

    def test_every_mode_and_chunker_returns_results(self):
        for mode in ("hybrid", "lexical", "dense"):
            for chunker in ("A-15s", "A-30s", "A-45s", "B", "B-prev", "C-512", "D"):
                results = self.run_search("sourdough starter fermentation", mode=mode, chunker=chunker).results
                self.assertTrue(results, (mode, chunker))
                self.assertEqual(results[0]["file_id"], "cooking_show.wav", (mode, chunker))

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

    def test_results_are_pinpointed_and_inside_their_chunk(self):
        for chunker in ("A-45s", "D"):
            for r in self.run_search("configuration plain text file edit by hand", chunker=chunker).results:
                self.assertLessEqual(r["end_seconds"] - r["start_seconds"], 20.0 + 1e-6)
                self.assertGreaterEqual(r["start_seconds"], r["chunk_start_seconds"] - 20.0)

    def test_no_overlapping_results_within_a_file(self):
        results = self.run_search("theme zebra plain dark colors window", chunker="A-15s", top_k=10).results
        linux = sorted((r["start_seconds"], r["end_seconds"]) for r in results if r["file_id"] == "linux_talk.wav")
        for (_, end), (start, _) in zip(linux, linux[1:]):
            self.assertLessEqual(end, start)

    def test_highlight_marks_stemmed_matches_and_escapes_html(self):
        top = self.run_search("compositors", mode="lexical").results[0]
        self.assertIn("<mark>compositor.</mark>".replace(".", ""), top["highlight"].replace(".", ""))
        self.assertNotIn("<script>", top["highlight"])

    def test_reranker_reorders_shortlist(self):
        query = "went back to plain dark colors afterwards"
        options = {"chunker": "B", "context": False, "dedupe_gap_seconds": 0}
        plain = self.run_search(query, **options).results
        engine = self.engine(rerankers={"bge-reranker": KeywordReranker("zebra")})
        reranked = self.run_search(query, engine=engine, reranker="bge-reranker", **options).results
        self.assertIn("zebra", reranked[0]["text"].lower())
        self.assertNotEqual([r["start_seconds"] for r in plain], [r["start_seconds"] for r in reranked])

    def test_reranker_ignored_outside_hybrid(self):
        engine = self.engine(rerankers={"bge-reranker": KeywordReranker("zebra")})
        for mode in ("lexical", "dense"):
            plain = self.run_search("theme plain dark", mode=mode, chunker="B").results
            with_reranker = self.run_search("theme plain dark", mode=mode, chunker="B", engine=engine, reranker="bge-reranker")
            self.assertNotIn("rerank", with_reranker.timings_ms)
            self.assertEqual(plain, with_reranker.results)

    def test_reranker_reads_context_only_when_context_is_on(self):
        spy = SpyReranker()
        engine = self.engine(rerankers={"bge-reranker": spy})
        query = "what desktop setup do you run at home these days"
        self.run_search(query, chunker="B-prev", context=True, engine=engine, reranker="bge-reranker")
        with_context = [p for call in spy.calls for p in call]
        spy.calls.clear()
        self.run_search(query, chunker="B-prev", context=False, engine=engine, reranker="bge-reranker")
        without_context = [p for call in spy.calls for p in call]
        self.assertTrue(any("desktop are you running" in p and "\n\n" in p for p in with_context))
        self.assertTrue(all("\n\n" not in p for p in without_context))

    def test_short_keyword_query_is_tightened_to_the_matched_word(self):
        top = self.run_search("Hyprland", mode="lexical", chunker="D").results[0]
        self.assertIn("Hyperland", top["text"])
        self.assertLessEqual(top["end_seconds"] - top["start_seconds"], 6.0)
        untightened = self.run_search("Hyprland", mode="lexical", chunker="D", keyword_span_padding_seconds=0).results[0]
        self.assertGreater(untightened["end_seconds"] - untightened["start_seconds"], top["end_seconds"] - top["start_seconds"])

    def test_long_queries_and_dense_mode_are_not_tightened(self):
        for mode, query in (("hybrid", "tiling window compositor for the desktop"), ("dense", "Hyprland")):
            loose = self.run_search(query, mode=mode, chunker="D", keyword_span_padding_seconds=0).results
            tight = self.run_search(query, mode=mode, chunker="D").results
            self.assertEqual([(r["start_seconds"], r["end_seconds"]) for r in loose],
                             [(r["start_seconds"], r["end_seconds"]) for r in tight], mode)

    def test_adjacent_sentences_are_merged_into_one_result(self):
        query = "Hyperland tiling Wayland compositor window snap mouse configuration plain text file edit hand"
        one_sentence = {"mode": "lexical", "chunker": "B", "top_k": 10, "span_max_sentences": 1, "span_min_seconds": 0}
        merged = self.run_search(query, **one_sentence).results
        separate = self.run_search(query, dedupe_gap_seconds=0, **one_sentence).results
        self.assertLess(len([r for r in merged if r["file_id"] == "linux_talk.wav"]),
                        len([r for r in separate if r["file_id"] == "linux_talk.wav"]))
        for r in merged:
            self.assertLessEqual(r["end_seconds"] - r["start_seconds"], 20.0 + 1e-6)

    def test_workspace_filter(self):
        response = SearchEngine(self.conn, embedders={"bge-small": HashEmbedder()}).search(
            "sourdough", "hybrid", 5, SearchConfig(**self.BASE), workspaces=("somewhere-else",)
        )
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

    def test_convex_fusion_runs(self):
        self.assertEqual(self.run_search("sourdough bread", fusion="convex").results[0]["file_id"], "cooking_show.wav")


class TestSearchConfig(unittest.TestCase):

    def test_frozen_defaults(self):
        config = SearchConfig()
        self.assertEqual((config.chunker, config.model, config.context, config.fusion, config.reranker),
                         ("A-30s", "gemma", False, "convex", None))
        self.assertEqual(config.weights, {"bm25": 1.0, "trigram": 1.0, "dense": 2.0})
        self.assertEqual((config.embedding_variant, config.dedupe_gap_seconds), ("gemma", 0.0))

    def test_embedding_variant(self):
        self.assertEqual(SearchConfig(chunker="A-30s", model="bge-small", context=True).embedding_variant, "bge-small+ctx")
        self.assertEqual(SearchConfig(chunker="B-prev", model="gemma", context=True).embedding_variant, "gemma+ctx")
        self.assertFalse(SearchConfig(chunker="C-512", context=True).uses_context)
        self.assertEqual(SearchConfig(chunker="A-30s", model="bge-small", context=False).embedding_variant, "bge-small")
        self.assertEqual(SearchConfig(chunker="D", model="gemma").embedding_variant, "gemma")

    def test_validation(self):
        for bad in ({"chunker": "E"}, {"model": "gpt"}, {"fusion": "max"}, {"reranker": "nope"}):
            with self.assertRaises(ValueError):
                SearchConfig(**bad)
        with self.assertRaisesRegex(ValueError, "unknown search config key"):
            SearchConfig.from_dict({"chunkr": "B"})

    def test_from_dict(self):
        config = SearchConfig.from_dict({"chunker": "B", "reranker": "bge-reranker", "dense_weight": 2})
        self.assertEqual((config.chunker, config.reranker, config.weights["dense"]), ("B", "bge-reranker", 2))


class TestSearchProvider(unittest.TestCase):

    def test_mock_mode_returns_empty_baseline(self):
        with mock.patch.dict(os.environ, {"EVAL_MOCK_MODE": "1"}):
            out = search_provider.call_api("q", {"config": {"mode": "dense", "top_k": 3}}, {})
        self.assertEqual(out["output"]["results"], [])
        self.assertEqual(out["output"]["status"], "mock_baseline")

    def test_passes_mode_top_k_and_config_overrides(self):
        with mock.patch.dict(os.environ, {"EVAL_MOCK_MODE": ""}), \
                mock.patch("src.search.engine.search", return_value=[{"file_id": "x.wav"}]) as search:
            out = search_provider.call_api(" q ", {"config": {"mode": "lexical", "top_k": 7, "chunker": "B", "basePath": "/repo"}}, {})
        kwargs = search.call_args.kwargs
        self.assertEqual((kwargs["query"], kwargs["mode"], kwargs["top_k"]), ("q", "lexical", 7))
        self.assertEqual(kwargs["config"].chunker, "B")
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
            evaluate_recall.run_benchmark(split="dev", modes=["dense"], search_config={"chunker": "B", "mode": "ignored"})
        self.assertEqual(call.call_args.kwargs["options"]["config"], {"chunker": "B", "mode": "dense", "top_k": 5})

    def test_bad_config_is_an_error(self):
        with mock.patch.dict(os.environ, {"EVAL_MOCK_MODE": ""}):
            out = search_provider.call_api("q", {"config": {"chunkr": "B"}}, {})
        self.assertIn("unknown search config key", out["error"])


if __name__ == "__main__":
    unittest.main()
