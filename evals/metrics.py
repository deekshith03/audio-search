"""
Retrieval & Evaluation Metrics for Conversational Audio Search.

Implements objective, code-based evaluation metrics:
- Temporal IoU (Intersection over Union)
- Strict Temporal Matching with positive overlap, duration bounds, and coverage
- Speaker Attribution Match
- Recall@k, HitRate@k, and MRR (Mean Reciprocal Rank)
"""

from typing import List, Dict, Any, Tuple

try:
    from speakers import resolve_result_speaker
except ImportError:
    from evals.speakers import resolve_result_speaker


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


def is_temporal_match(
    pred_start: float,
    pred_end: float,
    gt_start: float,
    gt_end: float,
    tolerance_seconds: float = 5.0,
    min_iou: float = 0.3,
    min_coverage: float = 0.25,
    max_duration: float = 120.0
) -> Tuple[bool, float, float]:
    """
    Strict temporal matching:
    1. Valid intervals: start < end
    2. Predicted duration is reasonable (<= max_duration)
    3. Positive overlap: overlap > 0
    4. Either IoU >= min_iou OR (start_delta <= tolerance AND overlap / pred_duration >= min_coverage)
    """
    if pred_end <= pred_start or gt_end <= gt_start:
        return False, abs(pred_start - gt_start), 0.0

    pred_dur = pred_end - pred_start
    if pred_dur > max_duration:
        return False, abs(pred_start - gt_start), 0.0

    overlap = min(pred_end, gt_end) - max(pred_start, gt_start)
    if overlap <= 0:
        return False, abs(pred_start - gt_start), 0.0

    iou = compute_temporal_iou(pred_start, pred_end, gt_start, gt_end)
    start_delta = abs(pred_start - gt_start)

    if iou >= min_iou:
        return True, start_delta, iou

    coverage = overlap / pred_dur
    is_match = (start_delta <= tolerance_seconds) and (coverage >= min_coverage)
    return is_match, start_delta, iou


def result_matches_moment(
    res: Dict[str, Any],
    gt: Dict[str, Any],
    tolerance_seconds: float = 5.0,
    min_iou: float = 0.3,
) -> bool:
    """A result finds a moment if it matches the moment or one of its equally good `alternatives`."""
    return any(_matches_one(res, m, tolerance_seconds, min_iou) for m in (gt, *gt.get("alternatives", ())))


def _matches_one(res: Dict[str, Any], gt: Dict[str, Any], tolerance_seconds: float, min_iou: float) -> bool:
    """File, speaker (resolved to a human name) and strict temporal match between one result and one moment."""
    if res.get("file_id") != gt.get("file_id"):
        return False
    gt_speaker = gt.get("speaker")
    if gt_speaker and resolve_result_speaker(res) != gt_speaker:
        return False
    pred_start = float(res.get("start_seconds", 0.0))
    pred_end = float(res.get("end_seconds", pred_start + 1.0))
    gt_start = float(gt.get("start_seconds", 0.0))
    gt_end = float(gt.get("end_seconds", gt_start + 1.0))
    match, _, _ = is_temporal_match(pred_start, pred_end, gt_start, gt_end, tolerance_seconds, min_iou)
    return match


def evaluate_retrieval(
    retrieved_results: List[Dict[str, Any]],
    expected_moments: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5],
    tolerance_seconds: float = 5.0,
    min_iou: float = 0.3,
    any_of: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates a retrieved list of segments against expected ground-truth moments.

    Recall@k is the fraction of expected moments found in the top k. With any_of=True (short_keyword
    queries, where each listed moment is an occurrence of the same term) it is 1.0 if any is found.
    MRR uses the rank of the first result matching any expected moment.
    """
    if not expected_moments:
        return {**{f"recall@{k}": 1.0 for k in k_values}, "mrr": 1.0}
    if not retrieved_results:
        return {**{f"recall@{k}": 0.0 for k in k_values}, "mrr": 0.0}

    def matches(res, gt):
        return result_matches_moment(res, gt, tolerance_seconds, min_iou)

    first_hit_rank = next(
        (rank for rank, res in enumerate(retrieved_results, start=1) if any(matches(res, gt) for gt in expected_moments)),
        None,
    )
    metrics: Dict[str, Any] = {"mrr": 1.0 / first_hit_rank if first_hit_rank else 0.0}
    for k in k_values:
        found = sum(1 for gt in expected_moments if any(matches(res, gt) for res in retrieved_results[:k]))
        metrics[f"recall@{k}"] = (1.0 if found else 0.0) if any_of else found / len(expected_moments)
    return metrics
