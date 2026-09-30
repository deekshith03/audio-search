"""
Benchmark query sets (qrels) and their split rules.

- dev: every tuning decision is made on this split.
- blind: 40 queries written after all tuning (frozen as "test2"), drafted mostly on the turns of the
  retired test and holdout splits by an agent that saw only those turns; every query was screened
  against all transcripts for equally good answers (`alternatives`), and main answers follow the
  right answer even on dev turns (4 do). Run once (evaluate_recall refuses a rerun).
- test and holdout (retired): each run once with the frozen configuration and kept in
  dataset/qrels/retired/ with their results; blind reuses their turns.

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
    "blind": os.path.join(ROOT, "dataset", "qrels", "blind_queries.json"),
}
RETIRED_SPLIT_PATHS = {
    "test": os.path.join(ROOT, "dataset", "qrels", "retired", "test_queries.json"),
    "holdout": os.path.join(ROOT, "dataset", "qrels", "retired", "holdout_queries.json"),
}
ONE_TIME_RESULTS = {"blind": os.path.join(ROOT, "evals", "results", "final_blind.json")}
TUNABLE_SPLITS = tuple(s for s in SPLIT_PATHS if s not in ONE_TIME_RESULTS)
MIN_KEYWORD_LABEL_SECONDS = 1.0
DEFAULT_SPLIT = "dev"
CATEGORIES = ("single_file", "multi_file", "near_miss", "short_keyword")
ANY_OF_CATEGORIES = {"short_keyword"}
EXPECTED_BREAKDOWN = {
    "dev": {"single_file": 33, "multi_file": 7, "near_miss": 24, "short_keyword": 9},
    "blind": {"single_file": 19, "multi_file": 5, "near_miss": 10, "short_keyword": 6},
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
        help="Query set to use. Defaults to dev so the blind set is only run deliberately.",
    )
