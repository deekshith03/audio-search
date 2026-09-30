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

import argparse
import json
import os
import sys
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import evaluate_retrieval, is_temporal_match, result_matches_moment  # noqa: E402
from qrels import ANY_OF_CATEGORIES, CATEGORIES, DEFAULT_SPLIT, ONE_TIME_RESULTS, add_split_argument, load_qrels  # noqa: E402
from search_provider import call_api  # noqa: E402


def top_result_is_hard_negative(top: Dict[str, Any], hard_negatives: List[Dict[str, Any]]) -> bool:
    for hn in hard_negatives:
        if top.get("file_id") != hn.get("file_id"):
            continue
        time_match, _, _ = is_temporal_match(
            top.get("start_seconds", 0.0), top.get("end_seconds", 0.0),
            hn.get("start_seconds", 0.0), hn.get("end_seconds", 0.0),
        )
        if time_match:
            return True
    return False


def run_benchmark(
    split: str = DEFAULT_SPLIT,
    modes: List[str] = ["hybrid", "lexical", "dense"],
    top_k: int = 5,
    search_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    queries = load_qrels(split)["queries"]
    summary_report = {}

    for mode in modes:
        mode_results = []
        cat_metrics = {c: [] for c in CATEGORIES}
        total_gt_moments = 0
        total_gt_recovered = {1: 0, 3: 0, 5: 0}

        for q in queries:
            cat = q["category"]
            expected = q["relevant_moments"]
            any_of = cat in ANY_OF_CATEGORIES

            provider_resp = call_api(
                prompt=q["query"],
                options={"config": {**(search_config or {}), "mode": mode, "top_k": top_k}},
                context={"vars": q}
            )
            if "error" in provider_resp:
                raise RuntimeError(f"{q['query_id']} [{mode}]: {provider_resp['error']}")
            retrieved = provider_resp.get("output", {}).get("results", [])

            query_eval = evaluate_retrieval(
                retrieved_results=retrieved,
                expected_moments=expected,
                k_values=[1, 3, 5],
                tolerance_seconds=5.0,
                min_iou=0.3,
                any_of=any_of,
            )

            # Micro recall counts ground-truth moments; a short_keyword query is one moment
            # (any of its listed occurrences), so repeated mentions do not inflate the total.
            total_gt_moments += 1 if any_of else len(expected)
            for k in (1, 3, 5):
                found = [any(result_matches_moment(res, exp) for res in retrieved[:k]) for exp in expected]
                total_gt_recovered[k] += (1 if any(found) else 0) if any_of else sum(found)

            has_positive_hit = query_eval.get("recall@5", 0.0) > 0.0
            avoided_hard_neg = bool(retrieved) and has_positive_hit and not top_result_is_hard_negative(retrieved[0], q.get("hard_negatives", []))

            query_eval.update({"avoided_hard_negative": avoided_hard_neg, "query_id": q["query_id"], "category": cat})
            mode_results.append(query_eval)
            cat_metrics[cat].append(query_eval)

        def avg(lst, key):
            return sum(item[key] for item in lst) / len(lst) if lst else 0.0

        near_miss = cat_metrics["near_miss"]
        summary_report[mode] = {
            "split": split,
            "macro_overall": {key: avg(mode_results, key) for key in ("recall@1", "recall@3", "recall@5", "mrr")},
            "micro_moments": {
                f"recall@{k}": total_gt_recovered[k] / total_gt_moments if total_gt_moments else 0.0 for k in (1, 3, 5)
            },
            "by_category": {
                cat: {key: avg(cat_metrics[cat], key) for key in ("recall@1", "recall@3", "recall@5", "mrr")}
                for cat in CATEGORIES
            },
            "near_miss_rejection_rate": (sum(r["avoided_hard_negative"] for r in near_miss) / len(near_miss)) if near_miss else 0.0,
            "queries": mode_results,
        }

    return summary_report


MODE_LABELS = {"hybrid": "HYBRID", "lexical": "LEXICAL (BM25)", "dense": "DENSE (VEC)"}
CATEGORY_LABELS = {"single_file": "Single-File", "multi_file": "Multi-File", "near_miss": "Near-Miss", "short_keyword": "Keyword"}


def print_scorecard(summary: Dict[str, Any]):
    split = next(iter(summary.values()), {}).get("split", "?")
    print("\n" + "=" * 82)
    print(f"        CONVERSATIONAL AUDIO HYBRID SEARCH: RECALL@K SCORECARD  [split: {split}]")
    print("=" * 82)
    print(f"{'Retrieval Mode':<16} | {'Recall@1':<10} | {'Recall@3':<10} | {'Recall@5':<10} | {'Micro R@5':<10} | {'MRR':<7}")
    print("-" * 82)
    for mode, data in summary.items():
        o = data["macro_overall"]
        print(f"{MODE_LABELS.get(mode, mode.upper()):<16} | {o['recall@1']:<10.2%} | {o['recall@3']:<10.2%} | {o['recall@5']:<10.2%} | {data['micro_moments']['recall@5']:<10.2%} | {o['mrr']:<7.3f}")
    print("=" * 82)

    print("\n[CATEGORY BREAKDOWN - MACRO RECALL@5]")
    header = f"{'Mode':<16} | " + " | ".join(f"{CATEGORY_LABELS[c]:<12}" for c in CATEGORIES) + f" | {'NM reject':<10}"
    print(header)
    print("-" * len(header))
    for mode, data in summary.items():
        c = data["by_category"]
        print(f"{MODE_LABELS.get(mode, mode.upper()):<16} | " + " | ".join(f"{c[cat]['recall@5']:<12.2%}" for cat in CATEGORIES) + f" | {data['near_miss_rejection_rate']:<10.2%}")
    print("=" * len(header) + "\n")


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


def refuse_rerun(split: str, force: bool) -> None:
    """A one-time split may be scored once; a forced rerun must be reported as a rerun."""
    path = ONE_TIME_RESULTS.get(split)
    if path and os.path.exists(path) and not force:
        sys.exit(f"The {split} split was already run ({path}). Pass --force only to report an explicit rerun.")


def save_one_time_result(split: str, report: Dict[str, Any], search_config: Dict[str, Any]) -> None:
    path = ONE_TIME_RESULTS.get(split)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"split": split, "search_config": search_config, "report": report}, f, indent=2)
        print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recall@k scorecard over a query split")
    add_split_argument(parser)
    parser.add_argument("--enforce-gate", action="store_true")
    parser.add_argument("--search-config", default="{}", help='JSON SearchConfig overrides, e.g. \'{"chunker": "B"}\'')
    parser.add_argument("--force", action="store_true", help="Rerun a one-time split (holdout) that already has results.")
    args = parser.parse_args()
    refuse_rerun(args.split, args.force)
    search_config = json.loads(args.search_config)
    report = run_benchmark(split=args.split, search_config=search_config)
    save_one_time_result(args.split, report, search_config)
    print_scorecard(report)
    if args.enforce_gate:
        check_gate(report)
