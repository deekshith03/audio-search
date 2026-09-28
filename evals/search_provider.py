"""
Search Provider Interface for Promptfoo Evaluation Harness.

Connects Promptfoo to the Conversational Audio Search Engine.
Supports 3 retrieval modes:
1. 'hybrid': Dual FTS + Dense Embeddings with Reciprocal Rank Fusion (RRF)
2. 'lexical': PostgreSQL Full-Text Search only
3. 'dense': pgvector Cosine Distance similarity only
"""

import os
import sys
import importlib
from typing import Dict, Any

# Add parent directory to path so search engine modules can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def call_api(prompt: str, options: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standard Promptfoo Python provider function.
    
    Args:
        prompt: The query string being evaluated.
        options: Provider options (e.g. {'mode': 'hybrid', 'top_k': 5}).
        context: Test context containing variables (e.g. expected_file, etc.).
        
    Returns:
        Dict with "output" containing retrieved search results list or explicit "error".
    """
    query = prompt.strip()
    config = options.get("config", {})
    mode = config.get("mode", "hybrid")
    top_k = int(config.get("top_k", 5))

    search_module = None
    try:
        search_module = importlib.import_module("search_engine")
    except ImportError:
        search_module = None

    if search_module and hasattr(search_module, "search"):
        results = search_module.search(query=query, mode=mode, top_k=top_k)
        return {
            "output": {
                "query": query,
                "mode": mode,
                "top_k": top_k,
                "results": results
            }
        }
    else:
        # If search engine is missing, fail-closed unless explicit mock mode is set
        if os.environ.get("EVAL_MOCK_MODE") == "1":
            return {
                "output": {
                    "query": query,
                    "mode": mode,
                    "top_k": top_k,
                    "status": "mock_baseline",
                    "results": []
                }
            }
        else:
            return {
                "error": "CRITICAL: search_engine.py is not yet implemented. Set EVAL_MOCK_MODE=1 to run pre-implementation baseline."
            }
