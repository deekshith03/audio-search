"""
Cross-encoder rerankers compared on the dev set (docs/PHASE_3_PLAN.md §4 "Reranker"). A reranker
reads the query and a candidate together, so it can separate look-alike passages that embeddings
place close together; it only reorders the fused shortlist.

Qwen3-Reranker is an instruction-tuned LLM judge: its inputs must follow the chat template from
its model card, or it scores near randomly (dev R@1 0.0 without the template).
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence

QWEN3_INSTRUCTION = "Given a search query, retrieve passages of a spoken conversation transcript that answer or discuss it"
QWEN3_QUERY_TEMPLATE = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    "<Instruct>: " + QWEN3_INSTRUCTION + "\n<Query>: {text}\n"
)
QWEN3_DOCUMENT_TEMPLATE = "<Document>: {text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


@dataclass(frozen=True)
class RerankerModel:
    key: str
    repo: str
    max_length: int
    query_template: str = "{text}"
    document_template: str = "{text}"


RERANKERS: Dict[str, RerankerModel] = {
    m.key: m
    for m in (
        RerankerModel("qwen3-reranker", "tomaarsen/Qwen3-Reranker-0.6B-seq-cls", 1024,
                      QWEN3_QUERY_TEMPLATE, QWEN3_DOCUMENT_TEMPLATE),
        RerankerModel("bge-reranker", "BAAI/bge-reranker-v2-m3", 1024),
        RerankerModel("minilm-reranker", "cross-encoder/ms-marco-MiniLM-L6-v2", 512),
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
        q = self.model.query_template.format(text=query)
        pairs = [(q, self.model.document_template.format(text=p)) for p in passages]
        scores = self._load().predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]
