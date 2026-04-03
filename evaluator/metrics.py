"""
Evaluation metrics for the RAG agent.

Metrics computed without needing a second LLM judge:

1. Retrieval Hit Rate       — did the query return ANY chunks?
2. Retrieval Precision@K    — how many of the top-K chunks are relevant?
3. Domain Gate Accuracy     — correct PASS/REJECT decision?
4. Answer Groundedness      — does the answer only use words found in context?
5. Citation Coverage        — fraction of answer sentences that have a citation
6. Semantic Similarity      — cosine sim between answer and expected answer
   (requires sentence-transformers, optional)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RetrievalMetrics:
    hit: bool                  # any chunk returned above threshold
    num_chunks: int            # number of chunks returned
    top_score: float           # highest cosine similarity score
    expected_sources: List[str]  # source names we expected to see
    found_sources: List[str]     # source names actually returned
    source_recall: float       # fraction of expected sources found


@dataclass
class DomainGateMetrics:
    expected_pass: bool        # ground truth: should this query be allowed?
    actual_pass: bool          # what the gate decided
    similarity_score: float    # raw cosine similarity
    correct: bool              # did the gate make the right call?


@dataclass
class AnswerMetrics:
    groundedness_score: float  # fraction of answer tokens found in context
    citation_coverage: float   # fraction of sentences that have a [DOC-X] citation
    semantic_similarity: Optional[float] = None  # vs expected answer (if provided)


@dataclass
class EvalResult:
    case_id: str
    query: str
    retrieval: Optional[RetrievalMetrics]
    domain_gate: DomainGateMetrics
    answer: Optional[AnswerMetrics]
    latency_ms: float
    notes: str = ""


def compute_retrieval_metrics(
    chunks: list,
    expected_source_names: List[str],
) -> RetrievalMetrics:
    found_names = [c.source_name for c in chunks]
    found_set = set(found_names)
    expected_set = set(expected_source_names)

    recall = (
        len(found_set & expected_set) / len(expected_set)
        if expected_set else 1.0
    )

    return RetrievalMetrics(
        hit=len(chunks) > 0,
        num_chunks=len(chunks),
        top_score=max((c.score for c in chunks), default=0.0),
        expected_sources=list(expected_set),
        found_sources=list(found_set),
        source_recall=recall,
    )


def compute_domain_gate_metrics(
    expected_pass: bool,
    actual_pass: bool,
    similarity_score: float,
) -> DomainGateMetrics:
    return DomainGateMetrics(
        expected_pass=expected_pass,
        actual_pass=actual_pass,
        similarity_score=similarity_score,
        correct=(expected_pass == actual_pass),
    )


def compute_answer_metrics(
    answer: str,
    context: str,
    expected_answer: Optional[str] = None,
    embedding_model=None,
) -> AnswerMetrics:
    groundedness = _groundedness(answer, context)
    citation_cov = _citation_coverage(answer)

    sem_sim = None
    if expected_answer and embedding_model is not None:
        import numpy as np
        vecs = embedding_model.encode(
            [answer, expected_answer], normalize_embeddings=True
        )
        sem_sim = float(np.dot(vecs[0], vecs[1]))

    return AnswerMetrics(
        groundedness_score=groundedness,
        citation_coverage=citation_cov,
        semantic_similarity=sem_sim,
    )


def _groundedness(answer: str, context: str) -> float:
    """Fraction of answer words that appear in the context (token overlap)."""
    answer_tokens = set(re.findall(r"\b\w+\b", answer.lower()))
    context_tokens = set(re.findall(r"\b\w+\b", context.lower()))

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "on",
        "at", "by", "for", "with", "as", "it", "its", "this", "that", "i",
        "we", "you", "he", "she", "they", "our", "your", "their", "and",
        "or", "but", "not", "no", "if", "so", "also", "here", "there",
    }
    content_tokens = answer_tokens - stopwords
    if not content_tokens:
        return 1.0

    overlap = content_tokens & context_tokens
    return len(overlap) / len(content_tokens)


def _citation_coverage(answer: str) -> float:
    """Fraction of sentences that contain at least one [DOC-X] citation."""
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
    if not sentences:
        return 0.0

    cited = sum(
        1 for s in sentences if re.search(r"\[[A-Z]+-[\w-]+\]", s)
    )
    return cited / len(sentences)


@dataclass
class AggregateMetrics:
    total_cases: int
    domain_gate_accuracy: float     # % of gate decisions that were correct
    retrieval_hit_rate: float       # % of in-domain queries that got chunks
    avg_source_recall: float        # avg fraction of expected sources found
    avg_top_retrieval_score: float  # avg highest chunk score
    avg_groundedness: float         # avg answer groundedness
    avg_citation_coverage: float    # avg citation coverage
    avg_semantic_similarity: Optional[float]  # avg semantic sim (if computed)
    avg_latency_ms: float
    results: List[EvalResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  AGENT EVALUATION REPORT",
            "=" * 60,
            f"  Total test cases       : {self.total_cases}",
            f"  Domain gate accuracy   : {self.domain_gate_accuracy:.1%}",
            f"  Retrieval hit rate     : {self.retrieval_hit_rate:.1%}",
            f"  Avg source recall      : {self.avg_source_recall:.1%}",
            f"  Avg top chunk score    : {self.avg_top_retrieval_score:.3f}",
            f"  Avg answer groundedness: {self.avg_groundedness:.1%}",
            f"  Avg citation coverage  : {self.avg_citation_coverage:.1%}",
        ]
        if self.avg_semantic_similarity is not None:
            lines.append(
                f"  Avg semantic similarity: {self.avg_semantic_similarity:.3f}"
            )
        lines += [
            f"  Avg latency            : {self.avg_latency_ms:.1f} ms",
            "=" * 60,
        ]
        return "\n".join(lines)


def aggregate(results: List[EvalResult]) -> AggregateMetrics:
    total = len(results)
    if total == 0:
        raise ValueError("No eval results to aggregate")

    gate_correct = sum(1 for r in results if r.domain_gate.correct)

    in_domain = [r for r in results if r.domain_gate.expected_pass]
    hits = [r for r in in_domain if r.retrieval and r.retrieval.hit]
    hit_rate = len(hits) / len(in_domain) if in_domain else 0.0

    recalls = [r.retrieval.source_recall for r in in_domain if r.retrieval]
    top_scores = [r.retrieval.top_score for r in in_domain if r.retrieval]

    answered = [r for r in results if r.answer is not None]
    groundedness = [r.answer.groundedness_score for r in answered]
    citations = [r.answer.citation_coverage for r in answered]
    sem_sims = [
        r.answer.semantic_similarity
        for r in answered
        if r.answer.semantic_similarity is not None
    ]

    return AggregateMetrics(
        total_cases=total,
        domain_gate_accuracy=gate_correct / total,
        retrieval_hit_rate=hit_rate,
        avg_source_recall=sum(recalls) / len(recalls) if recalls else 0.0,
        avg_top_retrieval_score=sum(top_scores) / len(top_scores) if top_scores else 0.0,
        avg_groundedness=sum(groundedness) / len(groundedness) if groundedness else 0.0,
        avg_citation_coverage=sum(citations) / len(citations) if citations else 0.0,
        avg_semantic_similarity=sum(sem_sims) / len(sem_sims) if sem_sims else None,
        avg_latency_ms=sum(r.latency_ms for r in results) / total,
        results=results,
    )
