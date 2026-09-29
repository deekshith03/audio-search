import os
import unittest
from unittest import mock

from huggingface_hub import try_to_load_from_cache

from src.search.rerankers import RERANKERS, Reranker

QUERY = "Why were arrays originally added to Postgres?"
RELEVANT = "Arrays were added to Postgres so the access control lists could be stored in a single column."
UNRELATED = "The starter needs days to mature and the dough needs a long cold proof."


def is_cached(repo: str) -> bool:
    return isinstance(try_to_load_from_cache(repo, "config.json"), str)


class TestRerankerRegistry(unittest.TestCase):

    def test_candidates(self):
        self.assertEqual(
            {k: m.repo for k, m in RERANKERS.items()},
            {"bge-reranker": "BAAI/bge-reranker-v2-m3"},
        )

    def test_lazy_and_empty_input(self):
        reranker = Reranker("bge-reranker")
        self.assertIsNone(reranker._ce)
        self.assertEqual(reranker.score("q", []), [])
        self.assertIsNone(reranker._ce)

    def test_score_pairs_query_with_each_passage(self):
        reranker = Reranker("bge-reranker")
        reranker._ce = mock.Mock(predict=mock.Mock(return_value=[0.5, 0.1]))
        self.assertEqual(reranker.score("q", ["d1", "d2"]), [0.5, 0.1])
        self.assertEqual(reranker._ce.predict.call_args.args[0], [("q", "d1"), ("q", "d2")])

    def test_unknown_key(self):
        with self.assertRaises(KeyError):
            Reranker("nope")


class TestRealRerankers(unittest.TestCase):
    """Loads each downloaded model once; skipped for models not in the local Hugging Face cache."""

    def test_relevant_passage_outranks_unrelated(self):
        for key, model in RERANKERS.items():
            with self.subTest(reranker=key):
                if not is_cached(model.repo):
                    self.skipTest(f"{model.repo} not downloaded")
                with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
                    scores = Reranker(key).score(QUERY, [UNRELATED, RELEVANT])
                self.assertEqual(len(scores), 2)
                self.assertGreater(scores[1], scores[0])


if __name__ == "__main__":
    unittest.main()
