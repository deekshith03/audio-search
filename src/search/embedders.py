"""
Local embedding model: EmbeddingGemma (google/embeddinggemma-300m, 768-d, gated on Hugging Face),
chosen on the dev set over bge-small, bge-base and Qwen3-Embedding (README §3.2).

Query and document prompts come from the model's own sentence-transformers config
(`encode_query` / `encode_document`). Vectors are L2-normalized, so cosine distance ranks them.
"""

import os
from dataclasses import dataclass
from typing import List, Sequence

from src.pipeline.common import load_env_file


@dataclass(frozen=True)
class EmbeddingModel:
    key: str
    repo: str
    dimensions: int
    max_tokens: int


EMBEDDING_MODEL = EmbeddingModel("gemma", "google/embeddinggemma-300m", 768, 2048)
BATCH_SIZE = 32


class Embedder:
    """Loads the model lazily, so importing this module or building one is cheap."""

    def __init__(self, model: EmbeddingModel = EMBEDDING_MODEL):
        self.model = model
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
