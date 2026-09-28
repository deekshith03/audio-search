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

    # Save clean standard YAML
    yaml_lines = [
        "description: 'Conversational Audio Hybrid Search: 18 Cross-File Evaluation Benchmark'",
        "prompts:",
        "  - '{{query}}'",
        "providers:",
        "  - id: python:evals/search_provider.py",
        "    label: Hybrid (RRF)",
        "    config:",
        "      mode: hybrid",
        "      top_k: 5",
        "  - id: python:evals/search_provider.py",
        "    label: Lexical (FTS)",
        "    config:",
        "      mode: lexical",
        "      top_k: 5",
        "  - id: python:evals/search_provider.py",
        "    label: Dense (pgvector)",
        "    config:",
        "      mode: dense",
        "      top_k: 5",
        "tests:"
    ]

    for t in tests:
        v = t["vars"]
        yaml_lines.append(f"  - description: \"{t['description']}\"")
        yaml_lines.append("    vars:")
        yaml_lines.append(f"      query: {json.dumps(v['query'], ensure_ascii=False)}")
        yaml_lines.append(f"      query_id: {v['query_id']}")
        yaml_lines.append(f"      category: {v['category']}")
        yaml_lines.append(f"      archetype: {v['archetype']}")
        yaml_lines.append("      expected_moments:")
        for em in v["expected_moments"]:
            yaml_lines.append("        - file_id: " + em["file_id"])
            yaml_lines.append(f"          turn_id: {em['turn_id']}")
            yaml_lines.append(f"          speaker: {json.dumps(em['speaker'], ensure_ascii=False)}")
            yaml_lines.append(f"          start_seconds: {em['start_seconds']}")
            yaml_lines.append(f"          end_seconds: {em['end_seconds']}")
            yaml_lines.append(f"          matched_text: {json.dumps(em['matched_text'], ensure_ascii=False)}")
        
        # Explicit empty list [] for empty hard_negatives to prevent YAML null coercion
        if v["hard_negatives"]:
            yaml_lines.append("      hard_negatives:")
            for hn in v["hard_negatives"]:
                yaml_lines.append("        - file_id: " + hn["file_id"])
                yaml_lines.append(f"          turn_id: {hn['turn_id']}")
                yaml_lines.append(f"          speaker: {json.dumps(hn['speaker'], ensure_ascii=False)}")
                yaml_lines.append(f"          start_seconds: {hn['start_seconds']}")
                yaml_lines.append(f"          end_seconds: {hn['end_seconds']}")
                yaml_lines.append(f"          reason: {json.dumps(hn['reason'], ensure_ascii=False)}")
        else:
            yaml_lines.append("      hard_negatives: []")

        yaml_lines.append("    assert:")
        yaml_lines.append("      - type: python")
        yaml_lines.append("        value: file://evals/assertions.py")

    with open("promptfooconfig.yaml", "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")

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
