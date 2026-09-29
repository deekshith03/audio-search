"""
Cross-encoder reranker (docs/PHASE_3_PLAN.md §4 "Reranker"). A reranker reads the query and a
candidate together, so it can separate look-alike passages that embeddings place close together;
it only reorders the fused shortlist.

bge-reranker-v2-m3 is the only candidate kept: on the dev grid, Qwen3-Reranker-0.6B was no better
at 2.5-7 s per search and ms-marco-MiniLM-L6-v2 lowered recall@1, so both were removed.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class RerankerModel:
    key: str
    repo: str
    max_length: int


RERANKERS: Dict[str, RerankerModel] = {
    m.key: m
    for m in (
        RerankerModel("bge-reranker", "BAAI/bge-reranker-v2-m3", 1024),
    )
}


class Reranker:
    """Loads the model lazily; `score` returns one relevance score per passage (higher is better)."""

    def __init__(self, key: str):
        self.model = RERANKERS[key]
        self._ce = None

    def _load(self):
        if self._ce is None:
            from sentence_transformers import CrossEncoder

            self._ce = CrossEncoder(self.model.repo, max_length=self.model.max_length)
        return self._ce

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        if not passages:
            return []
        scores = self._load().predict([(query, p) for p in passages], show_progress_bar=False)
        return [float(s) for s in scores]
