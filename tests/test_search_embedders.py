import unittest
from unittest import mock

import numpy as np

from src.search.embedders import EMBEDDING_MODEL, Embedder


class TestEmbedder(unittest.TestCase):

    def test_model_is_embeddinggemma(self):
        self.assertEqual((EMBEDDING_MODEL.key, EMBEDDING_MODEL.repo, EMBEDDING_MODEL.dimensions),
                         ("gemma", "google/embeddinggemma-300m", 768))

    def test_does_not_load_model_until_used(self):
        embedder = Embedder()
        self.assertIsNone(embedder._st)
        self.assertIs(embedder.model, EMBEDDING_MODEL)

    def test_encodes_with_model_prompts_and_normalization(self):
        embedder = Embedder()
        embedder._st = mock.Mock(encode_document=mock.Mock(return_value=np.array([[0.6, 0.8]])),
                                 encode_query=mock.Mock(return_value=np.array([[1.0, 0.0]])))
        self.assertEqual(embedder.encode_documents(["doc"]), [[0.6, 0.8]])
        self.assertEqual(embedder.encode_query("q"), [1.0, 0.0])
        self.assertTrue(embedder._st.encode_document.call_args.kwargs["normalize_embeddings"])
        self.assertTrue(embedder._st.encode_query.call_args.kwargs["normalize_embeddings"])


if __name__ == "__main__":
    unittest.main()
