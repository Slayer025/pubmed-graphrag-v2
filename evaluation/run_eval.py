#!/usr/bin/env python3
"""Run retrieval evaluation on the frozen query set.

This script evaluates the existing retrieval pipeline in either dense-only or
hybrid (dense + BM25 + RRF) mode.  It loads the pipeline via
``bootstrap_pipeline()``, runs retrieval for every question in
``queries.jsonl`` (no LLM generation), and reports Recall@5, Recall@10, and
MRR@10 against the expected PubMed article.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bootstrap.environment import configure_environment

# Ensure HuggingFace caches live in the platform temp directory so the script
# works on Windows as well as Linux/macOS.
os.environ.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "hf_cache"))
configure_environment()

from src.application.dto.search_config import SearchConfig
from src.bootstrap import bootstrap_pipeline
from src.bootstrap.bootstrap_artifacts import bootstrap_artifacts

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

QUERIES_PATH = Path(__file__).parent / "queries.jsonl"
DENSE_RESULTS_PATH = Path(__file__).parent / "results_dense_only.jsonl"
HYBRID_RESULTS_PATH = Path(__file__).parent / "results_hybrid.jsonl"
SUMMARY_PATH = Path(__file__).parent.parent / "outputs" / "retrieval_improvement_summary.json"


def _hybrid_results_path(rrf_k: int) -> Path:
    """Return the per-k hybrid result file path."""
    return Path(__file__).parent / f"results_hybrid_k{rrf_k}.jsonl"


def _load_queries(path: Path) -> list[dict]:
    """Load the frozen evaluation query set."""
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _evaluate_query(
    pipeline,
    query: dict,
    search_config: SearchConfig,
) -> dict:
    """Run retrieval for one query and compute per-query metrics."""
    question = str(query["question"])
    expected_article_id = str(query["expected_article_id"])

    start = time.perf_counter()
    results = pipeline.retrieve(question, search_config)
    latency_ms = (time.perf_counter() - start) * 1000

    top_10 = results[:10]
    correct_ranks = [
        rank
        for rank, result in enumerate(top_10, start=1)
        if str(result.article_id) == expected_article_id
    ]

    recall_at_5 = any(
        str(result.article_id) == expected_article_id for result in results[:5]
    )
    recall_at_10 = bool(correct_ranks)
    mrr_at_10 = 1.0 / correct_ranks[0] if correct_ranks else 0.0

    return {
        "query_id": str(query["query_id"]),
        "question": question,
        "expected_pubmed_id": str(query["expected_pubmed_id"]),
        "expected_article_id": expected_article_id,
        "recall@5": recall_at_5,
        "recall@10": recall_at_10,
        "mrr@10": round(mrr_at_10, 4),
        "latency_ms": round(latency_ms, 2),
        "num_results": len(results),
        "top_10": [
            {
                "rank": rank,
                "chunk_id": result.chunk_id,
                "article_id": str(result.article_id),
                "combined_score": round(result.combined_score, 4),
                "source": result.source,
            }
            for rank, result in enumerate(top_10, start=1)
        ],
    }


def _aggregate_metrics(records: list[dict]) -> dict:
    """Aggregate per-query results into summary metrics."""
    n = len(records)
    if n == 0:
        return {
            "num_queries": 0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "mrr@10": 0.0,
            "avg_latency_ms": 0.0,
        }
    return {
        "num_queries": n,
        "recall@5": round(sum(r["recall@5"] for r in records) / n, 4),
        "recall@10": round(sum(r["recall@10"] for r in records) / n, 4),
        "mrr@10": round(sum(r["mrr@10"] for r in records) / n, 4),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in records) / n, 2),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the PubMed GraphRAG retrieval pipeline.")
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Enable hybrid retrieval (dense + BM25 + RRF).",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF damping constant k (default: 60).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare existing dense-only and hybrid result files and print a summary.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run dense + hybrid for k=20,30,60 and print a tuning comparison.",
    )
    return parser.parse_args()


def _build_search_config(*, use_hybrid: bool, rrf_k: int = 60) -> SearchConfig:
    """Return the evaluation SearchConfig."""
    return SearchConfig(
        top_k=10,
        expand_depth=2,
        max_entity_degree=500,
        max_expansion_per_entity=100,
        max_expanded_nodes=2000,
        alpha=0.8,
        depth_scores=(1.0, 0.5, 0.25),
        max_results=20,
        use_hybrid=use_hybrid,
        rrf_k=rrf_k,
    )


def _run_evaluation(
    use_hybrid: bool,
    *,
    rrf_k: int = 60,
) -> tuple[Path, dict, list[dict]]:
    """Run one evaluation pass and return the output path, metrics, and details."""
    mode_label = "hybrid" if use_hybrid else "dense_only"
    if use_hybrid:
        results_path = _hybrid_results_path(rrf_k)
    else:
        results_path = DENSE_RESULTS_PATH
    search_config = _build_search_config(use_hybrid=use_hybrid, rrf_k=rrf_k)

    cache_dir = os.environ.get("ARTIFACT_CACHE_DIR", "").strip() or str(
        Path(tempfile.gettempdir()) / "pubmed-graphrag"
    )
    print(f"\nUsing artifact cache dir: {cache_dir}", flush=True)
    bootstrap_artifacts(cache_dir)

    pipeline = bootstrap_pipeline()

    queries = _load_queries(QUERIES_PATH)
    print(f"Loaded {len(queries)} queries from {QUERIES_PATH}", flush=True)
    print(f"Running {mode_label} evaluation (rrf_k={rrf_k})...", flush=True)

    detailed_results: list[dict] = []
    for query in queries:
        result = _evaluate_query(pipeline, query, search_config)
        detailed_results.append(result)
        logger.info(
            "%s | R@5=%s R@10=%s MRR@10=%s | latency=%s ms",
            result["query_id"],
            result["recall@5"],
            result["recall@10"],
            result["mrr@10"],
            result["latency_ms"],
        )

    metrics = _aggregate_metrics(detailed_results)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as handle:
        for record in detailed_results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    display_label = f"Hybrid RRF (k={rrf_k})" if use_hybrid else "Dense-only"
    print(f"\n{display_label} Retrieval Metrics", flush=True)
    print(f"  Queries evaluated: {metrics['num_queries']}", flush=True)
    print(f"  Recall@5:          {metrics['recall@5']}", flush=True)
    print(f"  Recall@10:         {metrics['recall@10']}", flush=True)
    print(f"  MRR@10:            {metrics['mrr@10']}", flush=True)
    print(f"  Avg latency:       {metrics['avg_latency_ms']} ms", flush=True)
    print(f"\nDetailed results saved to {results_path}", flush=True)

    return results_path, metrics, detailed_results


def _print_comparison_table(rows: list[tuple[str, dict]]) -> None:
    """Print a formatted comparison table in the terminal."""
    print("\n" + "=" * 70, flush=True)
    print("Retrieval Improvement Comparison", flush=True)
    print("=" * 70, flush=True)
    print(
        f"{'Mode':<18} | {'Recall@5':<10} | {'Recall@10':<11} | {'MRR@10':<10} | {'Avg Latency':<13}",
        flush=True,
    )
    print("-" * 70, flush=True)
    for label, metrics in rows:
        print(
            f"{label:<18} | "
            f"{metrics['recall@5']:<10} | "
            f"{metrics['recall@10']:<11} | "
            f"{metrics['mrr@10']:<10} | "
            f"{metrics['avg_latency_ms']:<13} ms",
            flush=True,
        )
    print("=" * 70, flush=True)


def _compute_deltas(dense_metrics: dict, hybrid_metrics: dict) -> dict:
    """Return absolute and relative improvements for each metric."""
    keys = ["recall@5", "recall@10", "mrr@10", "avg_latency_ms"]
    deltas: dict[str, dict[str, float]] = {}
    for key in keys:
        dense = dense_metrics[key]
        hybrid = hybrid_metrics[key]
        deltas[key] = {
            "dense": dense,
            "hybrid": hybrid,
            "absolute": round(hybrid - dense, 4),
            "relative": round((hybrid - dense) / dense, 4) if dense else None,
        }
    return deltas


def _save_summary(dense_metrics: dict, hybrid_metrics: dict) -> None:
    """Save the comparison summary to disk."""
    summary = {
        "metrics": {
            "dense_only": dense_metrics,
            "hybrid_rrf": hybrid_metrics,
        },
        "deltas": _compute_deltas(dense_metrics, hybrid_metrics),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {SUMMARY_PATH}", flush=True)


def _load_existing_metrics(path: Path) -> dict:
    """Load per-query records and aggregate into summary metrics."""
    with open(path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return _aggregate_metrics(records)


def _compare_existing() -> int:
    """Compare dense-only with the default hybrid result file."""
    if not DENSE_RESULTS_PATH.exists() or not HYBRID_RESULTS_PATH.exists():
        print(
            "Error: both results_dense_only.jsonl and results_hybrid.jsonl must exist "
            "before using --compare.",
            flush=True,
        )
        return 1
    dense_metrics = _load_existing_metrics(DENSE_RESULTS_PATH)
    hybrid_metrics = _load_existing_metrics(HYBRID_RESULTS_PATH)
    _print_comparison_table([("Dense-only", dense_metrics), ("Hybrid RRF", hybrid_metrics)])
    _save_summary(dense_metrics, hybrid_metrics)
    return 0


def _run_tuning() -> int:
    """Run dense + hybrid for k=20, 30, 60 and print a tuning comparison."""
    _, dense_metrics, _ = _run_evaluation(use_hybrid=False)
    hybrid_metrics_by_k: dict[int, dict] = {}
    for k in (20, 30, 60):
        _, hybrid_metrics, _ = _run_evaluation(use_hybrid=True, rrf_k=k)
        hybrid_metrics_by_k[k] = hybrid_metrics

    rows = [("Dense-only", dense_metrics)]
    for k in (20, 30, 60):
        rows.append((f"Hybrid RRF (k={k})", hybrid_metrics_by_k[k]))
    _print_comparison_table(rows)

    summary = {
        "metrics": {
            "dense_only": dense_metrics,
            **{f"hybrid_rrf_k{k}": hybrid_metrics_by_k[k] for k in (20, 30, 60)},
        },
        "deltas": {
            f"k{k}": _compute_deltas(dense_metrics, hybrid_metrics_by_k[k])
            for k in (20, 30, 60)
        },
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"\nTuning summary saved to {SUMMARY_PATH}", flush=True)
    return 0


def main() -> int:
    """Run one or both evaluations and print a comparison."""
    args = _parse_args()

    if args.tune:
        return _run_tuning()

    if args.compare:
        return _compare_existing()

    if args.hybrid:
        _run_evaluation(use_hybrid=True, rrf_k=args.rrf_k)
        return 0

    # Default: run dense-only evaluation.
    _, dense_metrics, _ = _run_evaluation(use_hybrid=False)
    print(
        "\nTip: re-run with --hybrid to produce hybrid results for a specific k.",
        flush=True,
    )
    print(
        f"\nAfter running --hybrid [--rrf-k N], compare with: "
        f"{Path(__file__).name} --compare",
        flush=True,
    )
    print(
        f"\nOr run the full tuning sweep with: {Path(__file__).name} --tune",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
