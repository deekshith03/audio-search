"""
95% bootstrap confidence intervals for micro recall@k, from a saved result file (read-only).

Whole queries are resampled with replacement (moments of one query are correlated, so resampling
moments independently would understate the interval); each draw's micro recall weights a query by
its number of moments, a short_keyword query counting as one, exactly as evaluate_recall does.

    uv run python evals/bootstrap_ci.py --split blind evals/results/blind_rerun_clean_index.json
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qrels import ANY_OF_CATEGORIES, RETIRED_SPLIT_PATHS, SPLIT_PATHS  # noqa: E402

DRAWS = 10_000
SEED = 0


def moment_weights(queries: Sequence[dict]) -> Dict[str, int]:
    return {q["query_id"]: 1 if q["category"] in ANY_OF_CATEGORIES else len(q["relevant_moments"]) for q in queries}


def micro_recall(per_query: Sequence[dict], weights: Dict[str, int], k: int) -> float:
    total = sum(weights[q["query_id"]] for q in per_query)
    return sum(q[f"recall@{k}"] * weights[q["query_id"]] for q in per_query) / total if total else 0.0


def bootstrap_ci(per_query: Sequence[dict], weights: Dict[str, int], k: int, draws: int = DRAWS, seed: int = SEED) -> Tuple[float, float]:
    rnd = random.Random(seed)
    values = sorted(micro_recall([rnd.choice(per_query) for _ in per_query], weights, k) for _ in range(draws))
    return values[int(0.025 * draws)], values[int(0.975 * draws) - 1]


def load_split_queries(split: str) -> List[dict]:
    path = SPLIT_PATHS.get(split) or RETIRED_SPLIT_PATHS.get(split)
    if path is None:
        raise SystemExit(f"Unknown split '{split}'")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["queries"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--split", required=True, help="Split whose qrels define the moment counts.")
    parser.add_argument("result", help="Result JSON with a 'report' of per-mode per-query recall.")
    args = parser.parse_args(argv)
    weights = moment_weights(load_split_queries(args.split))
    with open(args.result, "r", encoding="utf-8") as f:
        report = json.load(f)["report"]
    for mode, summary in report.items():
        per_query = summary["queries"]
        cells = []
        for k in (1, 5):
            lo, hi = bootstrap_ci(per_query, weights, k)
            cells.append(f"R@{k} {micro_recall(per_query, weights, k):.3f} ({lo:.2f}-{hi:.2f})")
        print(f"{mode:8} " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
