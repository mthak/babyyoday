#!/usr/bin/env python3
"""
Evaluate the agent against the eval dataset.

Usage:
    python evaluator/runner.py
    python evaluator/runner.py --config config.yaml --dataset evaluator/eval_dataset.yaml
    python evaluator/runner.py --output eval_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator.metrics import (
    EvalResult,
    aggregate,
    compute_answer_metrics,
    compute_domain_gate_metrics,
    compute_retrieval_metrics,
)
from inference.context_builder import build_context
from inference.domain_gate import DomainGate
from inference.retriever import Retriever

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset(dataset_path: str) -> list[dict]:
    with open(dataset_path) as f:
        data = yaml.safe_load(f)
    return data["cases"]


def run_evaluation(
    config: dict,
    cases: list[dict],
    use_semantic_similarity: bool = True,
) -> list[EvalResult]:
    retriever = Retriever(
        index_path=config["faiss"]["index_path"],
        metadata_path=config["faiss"]["metadata_path"],
        embedding_model_name=config["embedding"]["model_name"],
        top_k=config["retrieval"]["top_k"],
        relevance_threshold=config["retrieval"]["relevance_threshold"],
    )

    gate = DomainGate(
        centroid_path=config["domain_gate"]["centroid_path"],
        similarity_threshold=config["domain_gate"]["similarity_threshold"],
    )

    embedding_model = retriever.embedding_model if use_semantic_similarity else None

    results: list[EvalResult] = []

    for case in cases:
        case_id = case["id"]
        query = case["query"]
        expected_pass = case["expected_pass"]
        expected_sources = case.get("expected_sources", [])
        expected_answer = case.get("expected_answer")

        logger.info("[%s] %s", case_id, query)
        t0 = time.time()

        query_embedding = retriever.embed_query(query)
        actual_pass, similarity = gate.check(query_embedding)

        gate_metrics = compute_domain_gate_metrics(
            expected_pass=expected_pass,
            actual_pass=actual_pass,
            similarity_score=similarity,
        )

        retrieval_metrics = None
        answer_metrics = None

        if actual_pass:
            chunks = retriever.search(query)
            retrieval_metrics = compute_retrieval_metrics(chunks, expected_sources)

            if chunks:
                context = build_context(chunks)
                # In retrieval-only mode we evaluate groundedness against
                # a synthetic answer that references what was found.
                source_ids = [c.source_id for c in chunks]
                synthetic_answer = (
                    f"Based on our information: {context[:300]}... "
                    f"Sources: {', '.join(source_ids)}"
                )
                answer_metrics = compute_answer_metrics(
                    answer=synthetic_answer,
                    context=context,
                    expected_answer=expected_answer,
                    embedding_model=embedding_model,
                )

        latency_ms = (time.time() - t0) * 1000

        results.append(
            EvalResult(
                case_id=case_id,
                query=query,
                retrieval=retrieval_metrics,
                domain_gate=gate_metrics,
                answer=answer_metrics,
                latency_ms=latency_ms,
            )
        )

    return results


def print_case_table(results: list[EvalResult]):
    header = f"{'ID':<10} {'Gate':^6} {'Hit':^5} {'Recall':^8} {'Score':^7} {'Ground':^8} {'Latency':>8}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        gate_ok = "✓" if r.domain_gate.correct else "✗"
        hit = ("✓" if r.retrieval.hit else "✗") if r.retrieval else "—"
        recall = f"{r.retrieval.source_recall:.0%}" if r.retrieval else "—"
        top_score = f"{r.retrieval.top_score:.3f}" if r.retrieval else "—"
        ground = f"{r.answer.groundedness_score:.0%}" if r.answer else "—"
        latency = f"{r.latency_ms:.0f}ms"
        print(f"{r.case_id:<10} {gate_ok:^6} {hit:^5} {recall:^8} {top_score:^7} {ground:^8} {latency:>8}")
    print()


def print_failures(results: list[EvalResult]):
    failures = [r for r in results if not r.domain_gate.correct]
    if not failures:
        print("  No gate failures.\n")
        return
    for r in failures:
        expected = "PASS" if r.domain_gate.expected_pass else "REJECT"
        actual = "PASS" if r.domain_gate.actual_pass else "REJECT"
        print(f"  [{r.case_id}] Expected {expected}, got {actual} "
              f"(score={r.domain_gate.similarity_score:.3f})")
        print(f"    Query: {r.query}")
    print()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate the BabyYoday agent")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", default="evaluator/eval_dataset.yaml")
    parser.add_argument("--output", default=None, help="Save JSON report to this path")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Skip semantic similarity (faster)")
    args = parser.parse_args()

    config = load_config(args.config)
    cases = load_dataset(args.dataset)

    logger.info("Running %d evaluation cases ...", len(cases))
    results = run_evaluation(
        config=config,
        cases=cases,
        use_semantic_similarity=not args.no_semantic,
    )

    agg = aggregate(results)

    print_case_table(results)
    print(agg.summary())

    print("Gate failures:")
    print_failures(results)

    # Per-category breakdown
    categories: dict[str, list[EvalResult]] = {}
    for r in results:
        case = next(c for c in cases if c["id"] == r.case_id)
        cat = case.get("category", "unknown")
        categories.setdefault(cat, []).append(r)

    print("Per-category gate accuracy:")
    for cat, cat_results in sorted(categories.items()):
        correct = sum(1 for r in cat_results if r.domain_gate.correct)
        print(f"  {cat:<20} {correct}/{len(cat_results)} ({correct/len(cat_results):.0%})")
    print()

    if args.output:
        report = {
            "summary": {
                "total_cases": agg.total_cases,
                "domain_gate_accuracy": agg.domain_gate_accuracy,
                "retrieval_hit_rate": agg.retrieval_hit_rate,
                "avg_source_recall": agg.avg_source_recall,
                "avg_top_retrieval_score": agg.avg_top_retrieval_score,
                "avg_groundedness": agg.avg_groundedness,
                "avg_citation_coverage": agg.avg_citation_coverage,
                "avg_semantic_similarity": agg.avg_semantic_similarity,
                "avg_latency_ms": agg.avg_latency_ms,
            },
            "cases": [
                {
                    "id": r.case_id,
                    "query": r.query,
                    "gate_correct": r.domain_gate.correct,
                    "gate_similarity": r.domain_gate.similarity_score,
                    "retrieval_hit": r.retrieval.hit if r.retrieval else None,
                    "retrieval_chunks": r.retrieval.num_chunks if r.retrieval else None,
                    "source_recall": r.retrieval.source_recall if r.retrieval else None,
                    "top_score": r.retrieval.top_score if r.retrieval else None,
                    "groundedness": r.answer.groundedness_score if r.answer else None,
                    "semantic_similarity": r.answer.semantic_similarity if r.answer else None,
                    "latency_ms": r.latency_ms,
                }
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Report saved to %s", args.output)

    # Exit with error code if gate accuracy is below 80%
    if agg.domain_gate_accuracy < 0.80:
        logger.error(
            "Gate accuracy %.1f%% is below 80%% threshold — FAIL",
            agg.domain_gate_accuracy * 100,
        )
        sys.exit(1)

    logger.info("Evaluation passed.")


if __name__ == "__main__":
    main()
