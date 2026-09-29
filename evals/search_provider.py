"""
Search provider for the eval harness (promptfoo and evaluate_recall.py).

Calls `src.search.engine` in one of three retrieval modes:
1. 'hybrid':  BM25 + trigram + dense embeddings, fused (then optionally reranked)
2. 'lexical': keyword only (BM25 + trigram fuzzy matching)
3. 'dense':   pgvector cosine similarity only

Provider config keys other than `mode`, `top_k`, `workspaces` (and promptfoo's `basePath`) are `SearchConfig` overrides
(e.g. {"chunker": "B", "model": "gemma", "reranker": "bge-reranker"}), which is how the dev
grid varies the pipeline. Fails closed: any engine or database error is returned as an error,
never as an empty result list, unless EVAL_MOCK_MODE=1 asks for the empty baseline.
"""

import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROVIDER_KEYS = {"mode", "top_k", "workspaces", "basePath"}  # basePath is added by promptfoo


def call_api(prompt: str, options: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    query = prompt.strip()
    config = dict(options.get("config", {}))
    mode = config.get("mode", "hybrid")
    top_k = int(config.get("top_k", 5))
    output = {"query": query, "mode": mode, "top_k": top_k}

    if os.environ.get("EVAL_MOCK_MODE") == "1":
        return {"output": {**output, "status": "mock_baseline", "results": []}}

    try:
        from src.search.engine import DEFAULT_WORKSPACES, SearchConfig, search

        search_config = SearchConfig.from_dict({k: v for k, v in config.items() if k not in PROVIDER_KEYS})
        workspaces = tuple(config.get("workspaces") or DEFAULT_WORKSPACES)
        results = search(query=query, mode=mode, top_k=top_k, config=search_config, workspaces=workspaces)
    except Exception as e:
        return {"error": f"search failed ({type(e).__name__}): {e}"}
    return {"output": {**output, "results": results}}
