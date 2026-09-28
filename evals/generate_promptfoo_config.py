"""
Generates clean, readable promptfooconfig.json and promptfooconfig.yaml
directly from dataset/qrels/benchmark_queries.json using standard library.

Self-validates that parsed YAML structure matches parsed JSON 100%.
"""

import json
import os
import yaml


def generate_config():
    with open("dataset/qrels/benchmark_queries.json", "r", encoding="utf-8") as f:
        qrels_data = json.load(f)

    queries = qrels_data["queries"]

    providers = [
        {
            "id": "python:evals/search_provider.py",
            "label": "Hybrid (RRF)",
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
        "description": "Conversational Audio Hybrid Search: 18 Cross-File Evaluation Benchmark",
        "prompts": ["{{query}}"],
        "providers": providers,
        "tests": tests
    }

    # Save as promptfooconfig.json
    with open("promptfooconfig.json", "w", encoding="utf-8") as f:
        json.dump(promptfoo_config, f, indent=2, ensure_ascii=False)

    with open("promptfooconfig.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(promptfoo_config, f, sort_keys=False, allow_unicode=True, width=120)

    # True Self-Validation: parse both files and assert complete equality
    with open("promptfooconfig.json", "r", encoding="utf-8") as fj:
        d_json = json.load(fj)
    with open("promptfooconfig.yaml", "r", encoding="utf-8") as fy:
        d_yaml = yaml.safe_load(fy)

    assert d_json == d_yaml, "Self-validation failed: parsed promptfooconfig.yaml does not match promptfooconfig.json!"
    print(f"Generated promptfooconfig.yaml and promptfooconfig.json with {len(tests)} tests!")
    print("✓ Self-validation passed: promptfooconfig.yaml and promptfooconfig.json are 100% structurally identical.")


if __name__ == "__main__":
    generate_config()
