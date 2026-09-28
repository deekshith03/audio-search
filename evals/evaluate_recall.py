"""
Aggregate Recall@k and Retrieval Evaluation Runner.

Computes mathematically rigorous information retrieval metrics across the benchmark:
- Macro-Averaged Recall@k (Across queries)
- Micro-Averaged Recall@k (Across ground-truth moments)
- Recall@1 (Hit in top 1)
- Recall@3 (Hit in top 3)
- Recall@5 (Hit in top 5)
- Mean Reciprocal Rank (MRR)
- Near-Miss Precision (Paired success rate: positive in top-5 AND hard negative not at rank 1)

Enforces benchmark gate criteria when run with --enforce-gate:
- Hybrid Recall@5 >= 0.85
- Hybrid Recall@1 >= 0.60
- Hybrid strictly > Lexical and Dense baselines (Strict Ablation Proof)
"""

import os
import sys
import json
from typing import List, Dict, Any

from metrics import evaluate_retrieval, is_temporal_match
from search_provider import call_api


def run_benchmark(
    qrels_path: str = "dataset/qrels/benchmark_queries.json",
    modes: List[str] = ["hybrid", "lexical", "dense"],
    top_k: int = 5
) -> Dict[str, Any]:
    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels_data = json.load(f)

    queries = qrels_data["queries"]
    summary_report = {}

    for mode in modes:
        mode_results = []
        cat_metrics = {"single_file": [], "multi_file": [], "near_miss": []}

        total_gt_moments = 0
        total_gt_recovered = {1: 0, 3: 0, 5: 0}

        for q in queries:
            qid = q["query_id"]
            qtext = q["query"]
            cat = q["category"]
            expected = q["relevant_moments"]
            hard_negs = q.get("hard_negatives", [])
            total_gt_moments += len(expected)

            # Call search provider
            provider_resp = call_api(
                prompt=qtext,
                options={"config": {"mode": mode, "top_k": top_k}},
                context={"vars": q}
            )

            retrieved = provider_resp.get("output", {}).get("results", [])

            # Compute Recall@k and MRR
            query_eval = evaluate_retrieval(
                retrieved_results=retrieved,
                expected_moments=expected,
                k_values=[1, 3, 5],
                tolerance_seconds=5.0,
                min_iou=0.3
            )

            # Micro Recall: count moments recovered with BOTH temporal match AND speaker match
            for k in [1, 3, 5]:
                for exp in expected:
                    for res in retrieved[:k]:
                        file_match = (res.get("file_id") == exp.get("file_id"))
                        speaker_match = (res.get("speaker") == exp.get("speaker")) if exp.get("speaker") else True
                        time_match, _, _ = is_temporal_match(
                            res.get("start_seconds", 0.0),
                            res.get("end_seconds", 0.0),
                            exp.get("start_seconds", 0.0),
                            exp.get("end_seconds", 0.0)
                        )
                        if file_match and speaker_match and time_match:
                            total_gt_recovered[k] += 1
                            break

            # Near-Miss Precision:
            has_positive_hit = (query_eval.get("recall@5", 0.0) > 0.0)
            avoided_hard_neg = False

            if retrieved:
                top_1 = retrieved[0]
                is_hard_neg = False
                for hn in hard_negs:
                    time_match, _, _ = is_temporal_match(
                        top_1.get("start_seconds", 0.0),
                        top_1.get("end_seconds", 0.0),
                        hn.get("start_seconds", 0.0),
                        hn.get("end_seconds", 0.0)
                    )
                    file_match = (top_1.get("file_id") == hn.get("file_id"))
                    if file_match and time_match:
                        is_hard_neg = True
                        break
                
                avoided_hard_neg = (not is_hard_neg) and has_positive_hit

            query_eval["avoided_hard_negative"] = avoided_hard_neg
            query_eval["query_id"] = qid
            query_eval["category"] = cat

            mode_results.append(query_eval)
            cat_metrics[cat].append(query_eval)

        # Compute Macro-Averages
        def avg(lst, key):
            return sum(item[key] for item in lst) / len(lst) if lst else 0.0

        near_miss_total = len(cat_metrics["near_miss"])
        near_miss_success = sum(1 for r in cat_metrics["near_miss"] if r["avoided_hard_negative"])

        mode_summary = {
            "macro_overall": {
                "recall@1": avg(mode_results, "recall@1"),
                "recall@3": avg(mode_results, "recall@3"),
                "recall@5": avg(mode_results, "recall@5"),
                "mrr": avg(mode_results, "mrr"),
            },
            "micro_moments": {
                "recall@1": total_gt_recovered[1] / total_gt_moments if total_gt_moments else 0.0,
                "recall@3": total_gt_recovered[3] / total_gt_moments if total_gt_moments else 0.0,
                "recall@5": total_gt_recovered[5] / total_gt_moments if total_gt_moments else 0.0,
            },
            "by_category": {
                cat: {
                    "recall@1": avg(cat_metrics[cat], "recall@1"),
                    "recall@3": avg(cat_metrics[cat], "recall@3"),
                    "recall@5": avg(cat_metrics[cat], "recall@5"),
                    "mrr": avg(cat_metrics[cat], "mrr"),
                }
                for cat in cat_metrics
            },
            "near_miss_rejection_rate": (near_miss_success / near_miss_total) if near_miss_total > 0 else 0.0
        }

        summary_report[mode] = mode_summary

    return summary_report


def print_scorecard(summary: Dict[str, Any]):
    print("\n" + "=" * 82)
    print("        CONVERSATIONAL AUDIO HYBRID SEARCH: RECALL@K EVALUATION SCORECARD")
    print("=" * 82)
    print(f"{'Retrieval Mode':<16} | {'Recall@1':<10} | {'Recall@3':<10} | {'Recall@5':<10} | {'Micro R@5':<10} | {'MRR':<7}")
    print("-" * 82)
    for mode, data in summary.items():
        o = data["macro_overall"]
        m5 = data["micro_moments"]["recall@5"]
        mode_label = mode.upper()
        if mode == "hybrid":
            mode_label = "HYBRID (RRF)"
        elif mode == "lexical":
            mode_label = "LEXICAL (FTS)"
        elif mode == "dense":
            mode_label = "DENSE (VEC)"
            
        print(f"{mode_label:<16} | {o['recall@1']:<10.2%} | {o['recall@3']:<10.2%} | {o['recall@5']:<10.2%} | {m5:<10.2%} | {o['mrr']:<7.3f}")
    print("=" * 82)

    print("\n[CATEGORY BREAKDOWN - MACRO RECALL@5]")
    print(f"{'Mode':<16} | {'Single-File':<14} | {'Multi-File':<14} | {'Near-Miss':<14}")
    print("-" * 65)
    for mode, data in summary.items():
        c = data["by_category"]
        print(f"{mode.upper():<16} | {c['single_file']['recall@5']:<14.2%} | {c['multi_file']['recall@5']:<14.2%} | {c['near_miss']['recall@5']:<14.2%}")
    print("=" * 65 + "\n")


def check_gate(summary: Dict[str, Any]):
    """Strictly enforces blueprint threshold and ablation gates over micro moment recall."""
    import math

    print("[GATE VALIDATION CHECK]")
    failures = []

    # 1. Require all three retrieval modes
    for req_mode in ("hybrid", "lexical", "dense"):
        if req_mode not in summary:
            failures.append(f"Missing required retrieval mode '{req_mode}' in evaluation summary")

    if failures:
        print("❌ EVALUATION GATE FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)

    hybrid = summary["hybrid"]
    lexical = summary["lexical"]
    dense = summary["dense"]

    # 2. Validate micro_moments metrics existence and finite numbers
    def get_valid_metric(mode_data, mode_name, key):
        mm = mode_data.get("micro_moments")
        if not mm or key not in mm:
            failures.append(f"Missing '{key}' in micro_moments for {mode_name}")
            return None
        val = mm[key]
        if not isinstance(val, (int, float)) or isinstance(val, bool) or math.isnan(val) or math.isinf(val):
            failures.append(f"Non-numeric or NaN/Inf '{key}' in micro_moments for {mode_name}: {val}")
            return None
        return float(val)

    r5 = get_valid_metric(hybrid, "hybrid", "recall@5")
    r1 = get_valid_metric(hybrid, "hybrid", "recall@1")
    lex_r5 = get_valid_metric(lexical, "lexical", "recall@5")
    dense_r5 = get_valid_metric(dense, "dense", "recall@5")

    if failures or r5 is None or r1 is None or lex_r5 is None or dense_r5 is None:
        print("❌ EVALUATION GATE FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)

    # 3. Enforce thresholds and strict ablation
    if r5 < 0.85:
        failures.append(f"Hybrid Micro Recall@5 ({r5:.2%}) is below target threshold of 85.0%")
    if r1 < 0.60:
        failures.append(f"Hybrid Micro Recall@1 ({r1:.2%}) is below target threshold of 60.0%")
    if r5 <= lex_r5:
        failures.append(f"Strict Ablation Failure: Hybrid Micro Recall@5 ({r5:.2%}) is not strictly greater than Lexical ({lex_r5:.2%})")
    if r5 <= dense_r5:
        failures.append(f"Strict Ablation Failure: Hybrid Micro Recall@5 ({r5:.2%}) is not strictly greater than Dense ({dense_r5:.2%})")

    if failures:
        print("❌ EVALUATION GATE FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    else:
        print("✅ EVALUATION GATE PASSED: All retrieval thresholds and strict ablation criteria satisfied!")


if __name__ == "__main__":
    report = run_benchmark()
    print_scorecard(report)
    if "--enforce-gate" in sys.argv:
        check_gate(report)
