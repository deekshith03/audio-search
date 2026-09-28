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
            {
                "qwen3-reranker": "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
                "bge-reranker": "BAAI/bge-reranker-v2-m3",
                "minilm-reranker": "cross-encoder/ms-marco-MiniLM-L6-v2",
            },
        )

    def test_lazy_and_empty_input(self):
        reranker = Reranker("minilm-reranker")
        self.assertIsNone(reranker._ce)
        self.assertEqual(reranker.score("q", []), [])
        self.assertIsNone(reranker._ce)

    def test_qwen3_uses_its_chat_template_and_others_raw_text(self):
        qwen = RERANKERS["qwen3-reranker"]
        self.assertTrue(qwen.query_template.format(text="q").endswith("<Query>: q\n"))
        self.assertTrue(qwen.document_template.format(text="d").startswith("<Document>: d<|im_end|>"))
        self.assertEqual(RERANKERS["bge-reranker"].query_template.format(text="q"), "q")

    def test_score_formats_pairs(self):
        reranker = Reranker("qwen3-reranker")
        reranker._ce = mock.Mock(predict=mock.Mock(return_value=[0.5]))
        self.assertEqual(reranker.score("q", ["d"]), [0.5])
        (query, doc), = reranker._ce.predict.call_args.args[0]
        self.assertIn("<Query>: q", query)
        self.assertIn("<Document>: d", doc)

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
