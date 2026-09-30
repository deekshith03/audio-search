"""
Benchmark query sets (qrels) and their split rules.

- dev  (40%): every tuning decision is made on this split.
- test (60%): held out; run once with the frozen configuration.
- test2: 40 queries written after all tuning, drafted on turns no dev label used (mostly the spent
  test split's turns, plus the retired holdout's) by an agent that saw only those turns; every query
  was screened against all transcripts for equally good answers (`alternatives`), and main answers
  follow the right answer even on dev turns. Run once (evaluate_recall refuses a rerun).
- holdout (retired): 9 queries, run once on 2026-09-30; kept in dataset/qrels/retired/ with its
  result, replaced by test2, which reuses its turns.

Categories:
- single_file:   the answer is in exactly one recording
- multi_file:    the answer spans two or more recordings; every moment must be found
- near_miss:     one true answer plus hard negatives that must not rank first
- short_keyword: a query of up to three words (a term or name, e.g. "Thrilla in Manila"); finding any listed occurrence counts as a hit
"""

import argparse
import json
import os
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_PATHS = {
    "dev": os.path.join(ROOT, "dataset", "qrels", "dev_queries.json"),
    "test": os.path.join(ROOT, "dataset", "qrels", "test_queries.json"),
    "test2": os.path.join(ROOT, "dataset", "qrels", "test2_queries.json"),
}
RETIRED_SPLIT_PATHS = {"holdout": os.path.join(ROOT, "dataset", "qrels", "retired", "holdout_queries.json")}
ONE_TIME_RESULTS = {"test2": os.path.join(ROOT, "evals", "results", "final_test2.json")}
MIN_KEYWORD_LABEL_SECONDS = 1.0
DEFAULT_SPLIT = "dev"
CATEGORIES = ("single_file", "multi_file", "near_miss", "short_keyword")
ANY_OF_CATEGORIES = {"short_keyword"}
EXPECTED_BREAKDOWN = {
    "dev": {"single_file": 33, "multi_file": 7, "near_miss": 24, "short_keyword": 9},
    "test": {"single_file": 6, "multi_file": 6, "near_miss": 6, "short_keyword": 3},
    "test2": {"single_file": 19, "multi_file": 5, "near_miss": 10, "short_keyword": 6},
}
MAX_SHORT_KEYWORD_WORDS = 3  # the engine treats queries of up to 3 words as short keyword searches


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
