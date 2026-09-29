"""
Generates clean, readable promptfoo configs (YAML plus a structurally identical JSON copy)
for each query split: promptfooconfig.{dev,test}.{yaml,json}, from dataset/qrels/{split}_queries.json.

Self-validates that parsed YAML structure matches parsed JSON 100%.
"""

import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qrels import SPLIT_PATHS, load_qrels  # noqa: E402


def config_paths(split):
    return f"promptfooconfig.{split}.json", f"promptfooconfig.{split}.yaml"


def generate_config(split):
    queries = load_qrels(split)["queries"]

    providers = [
        {
            "id": "python:evals/search_provider.py",
            "label": "Hybrid",
            "config": {
                "mode": "hybrid",
                "top_k": 5
            }
        },
        {
            "id": "python:evals/search_provider.py",
            "label": "Lexical (FTS)",
            "config": {
                "mode": "lexical",
                "top_k": 5
            }
        },
        {
            "id": "python:evals/search_provider.py",
            "label": "Dense (pgvector)",
            "config": {
                "mode": "dense",
                "top_k": 5
            }
        }
    ]

    tests = []
    for q in queries:
        qid = q["query_id"]
        qtext = q["query"]
        category = q["category"]
        expected_moments = q["relevant_moments"]
        hard_negatives = q.get("hard_negatives", [])

        test_entry = {
            "description": f"[{qid}] {category.upper()}: {qtext[:50]}...",
            "vars": {
                "query": qtext,
                "query_id": qid,
                "category": category,
                "archetype": q["archetype"],
                "expected_moments": expected_moments,
                "hard_negatives": hard_negatives
            },
            "assert": [
                {
                    "type": "python",
                    "value": "file://evals/assertions.py"
                }
            ]
        }
        tests.append(test_entry)

    promptfoo_config = {
        "description": f"Conversational Audio Hybrid Search: {split} split ({len(tests)} queries)",
        "prompts": ["{{query}}"],
        "providers": providers,
        "tests": tests
    }

    # Save as promptfooconfig.json
    json_path, yaml_path = config_paths(split)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(promptfoo_config, f, indent=2, ensure_ascii=False)

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(promptfoo_config, f, sort_keys=False, allow_unicode=True, width=120)

    # True Self-Validation: parse both files and assert complete equality
    with open(json_path, "r", encoding="utf-8") as fj:
        d_json = json.load(fj)
    with open(yaml_path, "r", encoding="utf-8") as fy:
        d_yaml = yaml.safe_load(fy)

    assert d_json == d_yaml, f"Self-validation failed: {yaml_path} does not match {json_path}!"
    print(f"✓ [{split}] {yaml_path} + {json_path}: {len(tests)} tests, structurally identical.")


if __name__ == "__main__":
    for split in SPLIT_PATHS:
        generate_config(split)
