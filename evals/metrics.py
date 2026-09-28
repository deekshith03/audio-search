"""
Retrieval & Evaluation Metrics for Conversational Audio Search.

Implements objective, code-based evaluation metrics:
- Temporal IoU (Intersection over Union)
- Strict Temporal Matching with positive overlap, duration bounds, and coverage
- Speaker Attribution Match
- Recall@k, HitRate@k, and MRR (Mean Reciprocal Rank)
"""

from typing import List, Dict, Any, Tuple


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


def evaluate_retrieval(
    retrieved_results: List[Dict[str, Any]],
    expected_moments: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5],
    tolerance_seconds: float = 5.0,
    min_iou: float = 0.3
) -> Dict[str, Any]:
    """
    Evaluates a retrieved list of segments against expected ground-truth moments.
    """
    if not expected_moments:
        return {"recall@1": 1.0, "recall@3": 1.0, "recall@5": 1.0, "mrr": 1.0}

    if not retrieved_results:
        return {"recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0, "mrr": 0.0}

    first_hit_rank = None
    total_gt = len(expected_moments)

    # For MRR: rank of first retrieved result that matches ANY expected moment
    for rank, res in enumerate(retrieved_results, start=1):
        pred_file = res.get("file_id")
        pred_speaker = res.get("speaker")
        pred_start = float(res.get("start_seconds", 0.0))
        pred_end = float(res.get("end_seconds", pred_start + 1.0))

        for gt in expected_moments:
            gt_file = gt.get("file_id")
            gt_speaker = gt.get("speaker")
            gt_start = float(gt.get("start_seconds", 0.0))
            gt_end = float(gt.get("end_seconds", gt_start + 1.0))

            file_match = (pred_file == gt_file)
            speaker_match = (pred_speaker == gt_speaker) if gt_speaker else True
            match_time, _, _ = is_temporal_match(
                pred_start, pred_end, gt_start, gt_end, tolerance_seconds, min_iou
            )

            if file_match and speaker_match and match_time:
                if first_hit_rank is None:
                    first_hit_rank = rank
                break

    metrics = {
        "mrr": 1.0 / first_hit_rank if first_hit_rank else 0.0
    }

    # For Recall@k: fraction of expected moments recovered in top-k
    for k in k_values:
        top_k_found = 0
        for gt in expected_moments:
            found = False
            for res in retrieved_results[:k]:
                pred_file = res.get("file_id")
                pred_speaker = res.get("speaker")
                pred_start = float(res.get("start_seconds", 0.0))
                pred_end = float(res.get("end_seconds", pred_start + 1.0))

                gt_file = gt.get("file_id")
                gt_speaker = gt.get("speaker")
                gt_start = float(gt.get("start_seconds", 0.0))
                gt_end = float(gt.get("end_seconds", gt_start + 1.0))

                file_match = (pred_file == gt_file)
                speaker_match = (pred_speaker == gt_speaker) if gt_speaker else True
                match_time, _, _ = is_temporal_match(
                    pred_start, pred_end, gt_start, gt_end, tolerance_seconds, min_iou
                )
                if file_match and speaker_match and match_time:
                    found = True
                    break
            if found:
                top_k_found += 1

        metrics[f"recall@{k}"] = top_k_found / total_gt

    return metrics
