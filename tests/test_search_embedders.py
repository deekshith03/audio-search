import unittest

from src.search.embedders import DEFAULT_MODELS, MODELS, Embedder, model_of_variant, variant_key


class TestEmbedderRegistry(unittest.TestCase):

    def test_qwen3_registered_but_not_default(self):
        self.assertIn("qwen3", MODELS)
        self.assertEqual(DEFAULT_MODELS, ("bge-small", "bge-base", "gemma"))

    def test_dimensions(self):
        self.assertEqual({k: m.dimensions for k, m in MODELS.items()},
                         {"bge-small": 384, "bge-base": 768, "gemma": 768, "qwen3": 1024})

    def test_variant_keys_round_trip(self):
        self.assertEqual(variant_key("gemma", True), "gemma+ctx")
        self.assertEqual(variant_key("gemma", False), "gemma")
        self.assertIs(model_of_variant("gemma+ctx"), MODELS["gemma"])
        self.assertIs(model_of_variant("bge-small"), MODELS["bge-small"])

    def test_embedder_does_not_load_model_until_used(self):
        embedder = Embedder("bge-small")
        self.assertIsNone(embedder._st)
        self.assertEqual(embedder.model.repo, "BAAI/bge-small-en-v1.5")

    def test_unknown_model_rejected(self):
        with self.assertRaises(KeyError):
            Embedder("nope")


if __name__ == "__main__":
    unittest.main()
