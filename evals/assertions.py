"""
Pure Python Assertion Module for Promptfoo Evaluation Harness.

Replaces inlined JavaScript assertions with readable, testable Python code:
- Evaluates Strict Temporal Match:
    1. Strictly positive overlap (overlap > 0)
    2. Finite, valid intervals (start < end)
    3. (IoU >= 0.3) OR (start_delta <= 5.0s AND coverage >= 0.25 AND predicted_duration <= 120s)
- Evaluates Exact Speaker Attribution (anonymous SPEAKER_xx labels are resolved to names through
  dataset/speaker_labels/, the human labeling step, before comparing with qrel speakers)
- Evaluates Near-Miss Hard Negative Rejection at Rank #1
- Enforces Full Multi-File Recall for multi_file queries; any listed occurrence counts for
  short_keyword queries (same rule as the non-multi-file categories)
"""

from typing import Dict, Any, List

try:
    from speakers import LABELS_DIR, _speaker_names, resolve_result_speaker  # noqa: F401  (loaded by promptfoo from evals/)
except ImportError:
    from evals.speakers import LABELS_DIR, _speaker_names, resolve_result_speaker  # noqa: F401


def compute_temporal_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Computes Intersection over Union for two temporal intervals."""
    if end_a <= start_a or end_b <= start_b:
        return 0.0

    intersection_start = max(start_a, start_b)
    intersection_end = min(end_a, end_b)
    
    if intersection_end <= intersection_start:
        return 0.0
    
    intersection = intersection_end - intersection_start
    union = (end_a - start_a) + (end_b - start_b) - intersection
    return float(intersection / union) if union > 0 else 0.0


def check_temporal_match(
    pred_start: float,
    pred_end: float,
    gt_start: float,
    gt_end: float,
    tolerance_seconds: float = 5.0,
    min_iou: float = 0.3,
    min_coverage: float = 0.25,
    max_duration: float = 120.0
) -> bool:
    """
    Strict temporal matching:
    1. Valid intervals: start < end
    2. Predicted duration is reasonable (<= max_duration)
    3. Positive overlap: overlap > 0
    4. Either IoU >= min_iou OR (start_delta <= tolerance AND overlap / pred_duration >= min_coverage)
    """
    if pred_end <= pred_start or gt_end <= gt_start:
        return False

    pred_dur = pred_end - pred_start
    if pred_dur > max_duration:
        return False

    overlap = min(pred_end, gt_end) - max(pred_start, gt_start)
    if overlap <= 0:
        return False

    iou = compute_temporal_iou(pred_start, pred_end, gt_start, gt_end)
    if iou >= min_iou:
        return True

    start_delta = abs(pred_start - gt_start)
    coverage = overlap / pred_dur
    return (start_delta <= tolerance_seconds) and (coverage >= min_coverage)


def get_assert(output: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Primary Promptfoo Python assertion function.
    """
    results: List[Dict[str, Any]] = output.get("results", [])
    test_vars: Dict[str, Any] = context.get("vars", {})
    qid = test_vars.get("query_id", "UNKNOWN")
    category = test_vars.get("category", "single_file")
    expected_moments: List[Dict[str, Any]] = test_vars.get("expected_moments", [])
    hard_negatives: List[Dict[str, Any]] = test_vars.get("hard_negatives", [])

    if not results:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"[{qid}] FAILED: No search results returned from search engine."
        }

    # 1. Near-Miss Precision Check: Top-1 must NOT be a hard negative distractor
    if hard_negatives:
        top_res = results[0]
        top_file = top_res.get("file_id")
        top_start = float(top_res.get("start_seconds", 0.0))
        top_end = float(top_res.get("end_seconds", top_start + 1.0))

        for hn in hard_negatives:
            hn_file = hn.get("file_id")
            hn_start = float(hn.get("start_seconds", 0.0))
            hn_end = float(hn.get("end_seconds", hn_start + 1.0))

            if top_file == hn_file and check_temporal_match(top_start, top_end, hn_start, hn_end):
                return {
                    "pass": False,
                    "score": 0.0,
                    "reason": f"[{qid}] NEAR-MISS FAILURE: Rank #1 returned deceptive distractor from {hn_file}: {hn.get('reason', '')}"
                }

    # 2. Retrieval Recall Check: match expected moments
    hits = []
    for exp in expected_moments:
        exp_file = exp.get("file_id")
        exp_speaker = exp.get("speaker")
        exp_start = float(exp.get("start_seconds", 0.0))
        exp_end = float(exp.get("end_seconds", exp_start + 1.0))

        for rank, res in enumerate(results, start=1):
            r_file = res.get("file_id")
            r_speaker = resolve_result_speaker(res)
            r_start = float(res.get("start_seconds", 0.0))
            r_end = float(res.get("end_seconds", r_start + 1.0))

            file_match = (r_file == exp_file)
            speaker_match = (r_speaker == exp_speaker) if exp_speaker else True
            time_match = check_temporal_match(r_start, r_end, exp_start, exp_end)

            if file_match and speaker_match and time_match:
                hits.append({
                    "file_id": exp_file,
                    "speaker": exp_speaker,
                    "rank": rank,
                    "start_seconds": r_start
                })
                break

    if not hits:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"[{qid}] FAILED: None of the {len(expected_moments)} expected moments found in top {len(results)} results."
        }

    # For multi-file queries, require all expected moments to pass strictly
    score = len(hits) / len(expected_moments)
    if category == "multi_file":
        passed = (len(hits) == len(expected_moments))
    else:
        passed = (len(hits) > 0)

    return {
        "pass": passed,
        "score": score,
        "reason": f"[{qid}] {'PASSED' if passed else 'PARTIAL'}: Found {len(hits)}/{len(expected_moments)} moments (Top hit rank: {hits[0]['rank']})."
    }
