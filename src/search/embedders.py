"""
Local embedding models compared on the dev set. Qwen3 is registered but not embedded by default;
it is tried only if the first grid (the three cheaper models) suggests it is worth the cost.

Query and document prompts come from each model's own sentence-transformers config
(`encode_query` / `encode_document`), e.g. bge's "Represent this sentence for searching relevant
passages: " query instruction. Vectors are L2-normalized, so cosine distance ranks them.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

from src.pipeline.common import load_env_file


@dataclass(frozen=True)
class EmbeddingModel:
    key: str
    repo: str
    dimensions: int
    max_tokens: int


MODELS: Dict[str, EmbeddingModel] = {
    m.key: m
    for m in (
        EmbeddingModel("bge-small", "BAAI/bge-small-en-v1.5", 384, 512),
        EmbeddingModel("bge-base", "BAAI/bge-base-en-v1.5", 768, 512),
        EmbeddingModel("gemma", "google/embeddinggemma-300m", 768, 2048),
        EmbeddingModel("qwen3", "Qwen/Qwen3-Embedding-0.6B", 1024, 32768),
    )
}
DEFAULT_MODELS = ("bge-small", "bge-base", "gemma")
CONTEXT_SUFFIX = "+ctx"
BATCH_SIZE = 32


def variant_key(model_key: str, with_context: bool) -> str:
    return f"{model_key}{CONTEXT_SUFFIX}" if with_context else model_key


def model_of_variant(variant: str) -> EmbeddingModel:
    return MODELS[variant.removesuffix(CONTEXT_SUFFIX)]


class Embedder:
    """Loads the model lazily, so importing this module or building one is cheap."""

    def __init__(self, model_key: str):
        self.model = MODELS[model_key]
        self._st = None

    def _load(self):
        if self._st is None:
            from sentence_transformers import SentenceTransformer

            load_env_file()
            self._st = SentenceTransformer(self.model.repo, token=os.environ.get("HF_TOKEN") or None)
        return self._st

    def encode_documents(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._load().encode_document(
            list(texts), batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]

    def encode_query(self, text: str) -> List[float]:
        return self._load().encode_query([text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
