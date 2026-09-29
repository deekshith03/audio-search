"""
Dev-set tuning grid (docs/PHASE_3_PLAN.md §5). Comparison tooling only: deleted after the
configuration is frozen, with its result tables kept as findings in the write-up.

    uv run python evals/tune_grid.py --round families   # 11 chunk families, gemma, bge reranker, equal RRF
    uv run python evals/tune_grid.py --round joint      # families x 10 fusion settings x {no reranker, bge}
    uv run python evals/tune_grid.py --round span       # joint winner and runner-up family x span / dedupe settings

(The first round, 30 candidates over three embedding models, is kept in results/grid_models.json;
it fixed the model to EmbeddingGemma.) Each round writes evals/results/grid_{round}.json and prints
markdown tables; the joint round saves after every config and resumes where it stopped.

Selection rules were fixed before any results: hybrid micro recall@5 first; gaps of at most one
target moment are ties, broken by micro recall@1, then macro MRR, then the simpler/faster option.
In the joint round only configs with search p50 <= MAX_SEARCH_P50_MS are eligible, and a winning
reranker must beat the same config without one (keep rule), else its no-reranker twin wins.
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluate_recall  # noqa: E402
from metrics import result_matches_moment  # noqa: E402
from qrels import ANY_OF_CATEGORIES, load_qrels  # noqa: E402
from search_provider import call_api  # noqa: E402

SPLIT = "dev"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

GRID_MODEL = "gemma"
DEFAULT_RERANKER = "bge-reranker"
FAMILIES = ("A-15s", "A-15s+ctx", "A-30s", "A-30s+ctx", "A-45s", "A-45s+ctx", "B", "B+ctx", "B-prev+ctx", "C-512", "D")
JOINT_FAMILIES = ("A-15s", "A-30s", "A-30s+ctx", "B", "B-prev+ctx", "C-512")
RERANKER_OPTIONS = (None, "bge-reranker")
FUSION_WEIGHTS = ((1.0, 1.0, 1.0), (1.0, 1.0, 2.0), (1.0, 1.0, 3.0), (2.0, 2.0, 1.0), (1.0, 0.5, 1.0))
FUSIONS = tuple((fusion, w) for fusion in ("rrf", "convex") for w in FUSION_WEIGHTS)
SPAN_OPTIONS = tuple(
    {"span_extend_ratio": r, "span_min_seconds": m, "dedupe_gap_seconds": g}
    for r in (0.6, 0.8, 1.0) for m in (4.0, 6.0) for g in (0.0, 2.0)
)
RERANKER_MAX_ADDED_MS = 1500.0
MAX_SEARCH_P50_MS = 1600.0
RECALL_CEILING_K = 20

MODEL_RANK = {"bge-small": 0, "bge-base": 1, "gemma": 2}
RERANKER_RANK = {None: 0, "bge-reranker": 1}


def family_config(family: str) -> Dict[str, Any]:
    chunker, _, suffix = family.partition("+")
    return {"chunker": chunker, "model": GRID_MODEL, "context": suffix == "ctx"}


def family_of(config: Dict[str, Any]) -> str:
    return config["chunker"] + ("+ctx" if config.get("context") else "")


def family_candidates() -> List[Dict[str, Any]]:
    return [{**family_config(f), "reranker": DEFAULT_RERANKER} for f in FAMILIES]


def joint_candidates(families: Sequence[str] = JOINT_FAMILIES) -> List[Dict[str, Any]]:
    """Reranker outermost, so each reranker model is loaded once and can be freed afterwards."""
    return [
        {**family_config(f), "fusion": fusion, "bm25_weight": w[0], "trigram_weight": w[1], "dense_weight": w[2], "reranker": rr}
        for rr in RERANKER_OPTIONS for f in families for fusion, w in FUSIONS
    ]


def label(config: Dict[str, Any]) -> str:
    parts = [config["chunker"], config["model"] + ("+ctx" if config.get("context") else "")]
    if config.get("fusion", "rrf") != "rrf" or "bm25_weight" in config:
        weights = "/".join(f"{config.get(k, 1.0):g}" for k in ("bm25_weight", "trigram_weight", "dense_weight"))
        parts.append(f"{config.get('fusion', 'rrf')}({weights})")
    parts.append(config.get("reranker") or "no-rerank")
    if "span_extend_ratio" in config:
        parts.append(f"span {config['span_extend_ratio']:g}/{config['span_min_seconds']:g}s gap {config['dedupe_gap_seconds']:g}s")
    return " · ".join(parts)


def total_moments(split: str = SPLIT) -> int:
    return sum(1 if q["category"] in ANY_OF_CATEGORIES else len(q["relevant_moments"]) for q in load_qrels(split)["queries"])


def timed_call_api(latencies: List[float]) -> Callable:
    def wrapper(**kwargs):
        started = time.perf_counter()
        response = call_api(**kwargs)
        latencies.append((time.perf_counter() - started) * 1000)
        return response
    return wrapper


def run_mode(config: Dict[str, Any], mode: str, split: str = SPLIT) -> Dict[str, Any]:
    latencies: List[float] = []
    original = evaluate_recall.call_api
    evaluate_recall.call_api = timed_call_api(latencies)
    try:
        summary = evaluate_recall.run_benchmark(split=split, modes=[mode], search_config=config)[mode]
    finally:
        evaluate_recall.call_api = original
    latencies.sort()
    return {
        "micro_r1": summary["micro_moments"]["recall@1"],
        "micro_r5": summary["micro_moments"]["recall@5"],
        "macro_r1": summary["macro_overall"]["recall@1"],
        "macro_r3": summary["macro_overall"]["recall@3"],
        "macro_r5": summary["macro_overall"]["recall@5"],
        "mrr": summary["macro_overall"]["mrr"],
        "near_miss_reject": summary["near_miss_rejection_rate"],
        "by_category_r5": {cat: v["recall@5"] for cat, v in summary["by_category"].items()},
        "misses": [q["query_id"] for q in summary["queries"] if q["recall@5"] < 1.0],
        "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else None,
        "latency_p95_ms": round(latencies[int(0.95 * (len(latencies) - 1))], 1) if latencies else None,
    }


def fused_recall_ceiling(config: Dict[str, Any], split: str = SPLIT, k: int = RECALL_CEILING_K) -> float:
    """Micro recall@k of the hybrid list without reranking: the most a reranker could recover."""
    found = total = 0
    for q in load_qrels(split)["queries"]:
        response = call_api(prompt=q["query"], options={"config": {**config, "reranker": None, "mode": "hybrid", "top_k": k}}, context={})
        if "error" in response:
            raise RuntimeError(f"{q['query_id']}: {response['error']}")
        results = response["output"]["results"]
        hits = [any(result_matches_moment(r, m) for r in results) for m in q["relevant_moments"]]
        if q["category"] in ANY_OF_CATEGORIES:
            found, total = found + int(any(hits)), total + 1
        else:
            found, total = found + sum(hits), total + len(hits)
    return found / total if total else 0.0


def simplicity(config: Dict[str, Any]) -> Tuple:
    return (
        RERANKER_RANK.get(config.get("reranker"), 9),
        MODEL_RANK.get(config["model"], 9),
        int(bool(config.get("context"))),
        0 if config.get("fusion", "rrf") == "rrf" else 1,
    )


def rank_rows(rows: Sequence[Dict[str, Any]], margin: float) -> List[Dict[str, Any]]:
    """Orders rows best first under the fixed selection rules.

    Tiers: rows within `margin` of the best micro R@5 are tied; among them, within `margin` of
    the best micro R@1; then highest MRR; then the simplest config. Repeats on the remainder.
    """
    remaining, ordered = list(rows), []
    while remaining:
        best_r5 = max(r["hybrid"]["micro_r5"] for r in remaining)
        tier = [r for r in remaining if r["hybrid"]["micro_r5"] >= best_r5 - margin - 1e-9]
        best_r1 = max(r["hybrid"]["micro_r1"] for r in tier)
        tier = [r for r in tier if r["hybrid"]["micro_r1"] >= best_r1 - margin - 1e-9]
        pick = min(tier, key=lambda r: (-round(r["hybrid"]["mrr"], 3), simplicity(r["config"])))
        ordered.append(pick)
        remaining.remove(pick)
    return ordered


def reranker_keep_decision(rows: Sequence[Dict[str, Any]], margin: float) -> Dict[str, Any]:
    """Keep rule: a reranker must beat no-reranker on R@1, MRR or near-miss rejection, not lose
    R@5 by more than the tie margin, and add at most RERANKER_MAX_ADDED_MS median latency."""
    baseline = next(r for r in rows if r["config"].get("reranker") is None)
    eligible = []
    for r in rows:
        if r is baseline:
            continue
        h, b = r["hybrid"], baseline["hybrid"]
        added_ms = (h["latency_p50_ms"] or 0.0) - (b["latency_p50_ms"] or 0.0)
        better = h["micro_r1"] > b["micro_r1"] + margin / 2 or h["mrr"] > b["mrr"] + 0.02 or h["near_miss_reject"] > b["near_miss_reject"]
        if better and h["micro_r5"] >= b["micro_r5"] - margin - 1e-9 and added_ms <= RERANKER_MAX_ADDED_MS:
            eligible.append(r)
    if not eligible:
        return baseline
    return rank_rows(eligible, margin)[0]


def evaluate(config: Dict[str, Any], with_diagnostics: bool, lexical_cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"config": config, "label": label(config), "hybrid": run_mode(config, "hybrid")}
    if with_diagnostics:
        if config["chunker"] not in lexical_cache:
            lexical_cache[config["chunker"]] = run_mode(config, "lexical")
        row["lexical"] = lexical_cache[config["chunker"]]
        row["dense"] = run_mode(config, "dense")
        row["fused_r20"] = fused_recall_ceiling(config)
    print(f"  {row['label']:<55} R@5 {row['hybrid']['micro_r5']:.3f}  R@1 {row['hybrid']['micro_r1']:.3f}  "
          f"MRR {row['hybrid']['mrr']:.3f}  p50 {row['hybrid']['latency_p50_ms']:.0f} ms", flush=True)
    return row


def results_path(round_name: str) -> str:
    return os.path.join(RESULTS_DIR, f"grid_{round_name}.json")


def load_round(round_name: str) -> Dict[str, Any]:
    path = results_path(round_name)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run --round {round_name} first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_round(round_name: str, rows: List[Dict[str, Any]], selected: List[Dict[str, Any]], note: str = "", complete: bool = True) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    doc = {
        "round": round_name, "split": SPLIT, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "complete": complete, "tie_margin": 1 / total_moments(), "note": note,
        "rows": rows, "selected": [r["config"] for r in selected],
    }
    with open(results_path(round_name), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def markdown_table(rows: Sequence[Dict[str, Any]], diagnostics: bool) -> str:
    head = "| # | Candidate | Micro R@5 | Micro R@1 | Macro R@5 | MRR | NM reject | p50 ms |"
    sep = "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |"
    if diagnostics:
        head += " Lexical R@5 | Dense R@5 | Fused R@20 |"
        sep += " ---: | ---: | ---: |"
    lines = [head, sep]
    for i, r in enumerate(rows, start=1):
        h = r["hybrid"]
        line = (f"| {i} | {r['label']} | {h['micro_r5']:.3f} | {h['micro_r1']:.3f} | {h['macro_r5']:.3f} | "
                f"{h['mrr']:.3f} | {h['near_miss_reject']:.2f} | {h['latency_p50_ms']:.0f} |")
        if diagnostics:
            line += f" {r['lexical']['micro_r5']:.3f} | {r['dense']['micro_r5']:.3f} | {r['fused_r20']:.3f} |"
        lines.append(line)
    return "\n".join(lines)


def is_eligible(row: Dict[str, Any]) -> bool:
    return (row["hybrid"]["latency_p50_ms"] or 0.0) <= MAX_SEARCH_P50_MS


def twin_without_reranker(rows: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    target = {**config, "reranker": None}
    return next((r for r in rows if r["config"] == target), None)


def select_joint(rows: Sequence[Dict[str, Any]], margin: float) -> Tuple[Dict[str, Any], str]:
    eligible = [r for r in rows if is_eligible(r)]
    if not eligible:
        raise SystemExit(f"no config has search p50 <= {MAX_SEARCH_P50_MS:.0f} ms")
    best = rank_rows(eligible, margin)[0]
    if best["config"].get("reranker") is None:
        return best, "best eligible config uses no reranker"
    twin = twin_without_reranker(rows, best["config"])
    if twin is None:
        return best, "no no-reranker twin to compare"
    chosen = reranker_keep_decision([twin, best], margin)
    if chosen is twin:
        return twin, f"keep rule rejected {best['config']['reranker']}; its no-reranker twin wins"
    return best, f"keep rule kept {best['config']['reranker']}"


def best_per_family(rows: Sequence[Dict[str, Any]], margin: float) -> List[Dict[str, Any]]:
    families: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        families.setdefault(family_of(r["config"]), []).append(r)
    return rank_rows([rank_rows(group, margin)[0] for group in families.values()], margin)


def span_leaders(joint: Dict[str, Any], margin: float) -> List[Dict[str, Any]]:
    """The joint winner plus the best eligible config from a different chunk family."""
    winner = joint["selected"][0]
    others = [r for r in joint["rows"] if is_eligible(r) and family_of(r["config"]) != family_of(winner)]
    return [winner] + ([rank_rows(others, margin)[0]["config"]] if others else [])


def evict_reranker(key: Optional[str]) -> None:
    from src.search.engine import default_engine

    if key:
        default_engine()._rerankers.pop(key, None)


def run_round(round_name: str, families: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    margin = 1 / total_moments()
    lexical_cache: Dict[str, Dict[str, Any]] = {}
    started = time.perf_counter()

    if round_name == "families":
        configs = family_candidates()
        print(f"Round families: {len(configs)} configs on {SPLIT} (tie margin {margin:.3f})", flush=True)
        rows = rank_rows([evaluate(c, True, lexical_cache) for c in configs], margin)
        save_round(round_name, rows, rows[:1])
        print(f"\n{markdown_table(rows, True)}\n")

    elif round_name == "joint":
        configs = joint_candidates(families or JOINT_FAMILIES)
        previous = load_round("joint") if os.path.exists(results_path("joint")) else {"rows": []}
        done = {json.dumps(r["config"], sort_keys=True): r for r in previous["rows"]}
        rows = [done[k] for k in (json.dumps(c, sort_keys=True) for c in configs) if k in done]
        print(f"Round joint: {len(configs)} configs, {len(rows)} already done (tie margin {margin:.3f})", flush=True)
        current_reranker = None
        for config in configs:
            if json.dumps(config, sort_keys=True) in done:
                continue
            if config["reranker"] != current_reranker:
                evict_reranker(current_reranker)
                current_reranker = config["reranker"]
            rows.append(evaluate(config, False, lexical_cache))
            save_round(round_name, rows, [], complete=False)
        evict_reranker(current_reranker)
        winner, note = select_joint(rows, margin)
        ordered = rank_rows(rows, margin)
        save_round(round_name, ordered, [winner], note)
        print(f"\nTop 25 of all configs (only search p50 <= {MAX_SEARCH_P50_MS:.0f} ms is eligible to win):\n{markdown_table(ordered[:25], False)}\n")
        print(f"Best per family:\n{markdown_table(best_per_family(rows, margin), False)}\n")
        print(f"selected: {winner['label']}  ({note})")
        rows = ordered

    elif round_name == "span":
        leaders = span_leaders(load_round("joint"), margin)
        configs = [{**leader, **span} for leader in leaders for span in SPAN_OPTIONS]
        print(f"Round span: {len(configs)} configs on {SPLIT} for {[label(c) for c in leaders]}", flush=True)
        rows = [evaluate(c, False, lexical_cache) for c in configs]
        eligible = [r for r in rows if is_eligible(r)]
        rows = rank_rows(rows, margin)
        selected = rank_rows(eligible, margin)[:1] if eligible else []
        save_round(round_name, rows, selected)
        print(f"\n{markdown_table(rows, False)}\n")
        print(f"Best per leader:\n{markdown_table(best_per_family(rows, margin), False)}\n")
        print(f"selected: {selected[0]['label'] if selected else 'none eligible'}")

    else:
        raise SystemExit("--round must be families, joint or span")

    print(f"round {round_name} took {time.perf_counter() - started:.0f}s; saved {results_path(round_name)}")
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Dev-set tuning grid.")
    parser.add_argument("--round", required=True, choices=["families", "joint", "span"])
    parser.add_argument("--families", help=f"Comma-separated families for the joint round (default {','.join(JOINT_FAMILIES)}).")
    args = parser.parse_args(argv)
    run_round(args.round, args.families.split(",") if args.families else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
