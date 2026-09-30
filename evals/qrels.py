"""
Benchmark query sets (qrels) and their split rules.

- dev  (40%): every tuning decision is made on this split.
- test (60%): held out; run once with the frozen configuration.
- holdout: written after the test run on turns no other split uses, before any post-test tuning,
  by an agent that saw only those turns; run once at the end (evaluate_recall refuses a rerun).

Categories:
- single_file:   the answer is in exactly one recording
- multi_file:    the answer spans two or more recordings; every moment must be found
- near_miss:     one true answer plus hard negatives that must not rank first
- short_keyword: a one- or two-word query; finding any listed occurrence counts as a hit
"""

import argparse
import json
import os
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_PATHS = {
    "dev": os.path.join(ROOT, "dataset", "qrels", "dev_queries.json"),
    "test": os.path.join(ROOT, "dataset", "qrels", "test_queries.json"),
    "holdout": os.path.join(ROOT, "dataset", "qrels", "holdout_queries.json"),
}
ONE_TIME_RESULTS = {"holdout": os.path.join(ROOT, "evals", "results", "final_holdout.json")}
DEFAULT_SPLIT = "dev"
CATEGORIES = ("single_file", "multi_file", "near_miss", "short_keyword")
ANY_OF_CATEGORIES = {"short_keyword"}
EXPECTED_BREAKDOWN = {
    "dev": {"single_file": 33, "multi_file": 7, "near_miss": 24, "short_keyword": 9},
    "test": {"single_file": 6, "multi_file": 6, "near_miss": 6, "short_keyword": 3},
    "holdout": {"single_file": 6, "multi_file": 0, "near_miss": 2, "short_keyword": 1},
}
MAX_SHORT_KEYWORD_WORDS = 2


def load_qrels(split: str = DEFAULT_SPLIT) -> Dict[str, Any]:
    if split not in SPLIT_PATHS:
        raise ValueError(f"Unknown split '{split}'. Expected one of {sorted(SPLIT_PATHS)}.")
    with open(SPLIT_PATHS[split], "r", encoding="utf-8") as f:
        return json.load(f)


def add_split_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split",
        choices=sorted(SPLIT_PATHS),
        default=DEFAULT_SPLIT,
        help="Query set to use. Defaults to dev so the held-out test set is only run deliberately.",
    )
