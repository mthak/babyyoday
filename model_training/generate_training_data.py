#!/usr/bin/env python3
"""
Auto-generate LoRA training data from the business's document chunks.

Produces a JSONL file where each line is a training example:
  {"system": "...", "user": "...", "assistant": "..."}

Two sources of training data:
  1. Auto-generated Q&A pairs from document chunks (no LLM needed)
  2. Manual Q&A pairs from a YAML file the business owner provides

Usage:
    python model_training/generate_training_data.py \
        --docs-dir ./data/docs/ \
        --output ./data/training_data.jsonl \
        --business-name "Sweet Rise Bakery" \
        --manual-qa evaluator/eval_dataset.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.chunker import process_directory
from inference.context_builder import build_context
from inference.prompt import format_training_example
from inference.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# ── Question templates keyed by detected content type ────────────────────────

QUESTION_TEMPLATES = {
    "price": [
        "How much does {item} cost?",
        "What is the price of {item}?",
        "What do you charge for {item}?",
    ],
    "hours": [
        "What are your opening hours?",
        "What time do you open?",
        "Are you open on weekends?",
        "What time do you close?",
    ],
    "policy": [
        "What is your policy on {topic}?",
        "How does {topic} work?",
        "Can you explain your {topic} policy?",
    ],
    "item": [
        "Do you have {item}?",
        "Tell me about {item}.",
        "What can you tell me about {item}?",
        "Is {item} available?",
    ],
    "general": [
        "What do you offer?",
        "Can you tell me more about this?",
        "What information do you have about this?",
    ],
}

# Patterns that signal what kind of content a chunk contains
PRICE_PATTERN = re.compile(r"\$[\d.,]+|\d+\s*(?:dollars|USD|GBP|EUR)", re.I)
HOURS_PATTERN = re.compile(r"\d{1,2}:\d{2}\s*(?:AM|PM)|monday|tuesday|wednesday|thursday|friday|saturday|sunday", re.I)
POLICY_PATTERN = re.compile(r"policy|cancell|refund|deposit|advance notice|require", re.I)
ITEM_PATTERN = re.compile(r"(?:—|•|-)\s*(.+?)(?:\s*—|\s*\$|\s*\n|$)", re.M)


def _detect_content_type(text: str) -> str:
    if PRICE_PATTERN.search(text):
        return "price"
    if HOURS_PATTERN.search(text):
        return "hours"
    if POLICY_PATTERN.search(text):
        return "policy"
    return "item"


def _extract_items(text: str) -> list[str]:
    """Extract bullet-point items or named things from a chunk."""
    matches = ITEM_PATTERN.findall(text)
    items = [m.strip().split("(")[0].strip() for m in matches if len(m.strip()) > 3]
    return items[:3]  # cap at 3 to avoid noise


def _extract_topics(text: str) -> list[str]:
    """Extract policy/topic keywords from a chunk."""
    topics = re.findall(r"(?:cancell\w+|refund\w*|deposit\w*|loyalty\w*|catering\w*|ordering\w*|custom\w*)", text, re.I)
    return list(dict.fromkeys(t.lower() for t in topics))[:3]


def generate_qa_from_chunk(
    text: str,
    source_id: str,
    source_name: str,
    business_name: str,
) -> list[dict[str, str]]:
    """Generate 1–3 Q&A training pairs from a single chunk."""
    content_type = _detect_content_type(text)
    examples: list[dict[str, str]] = []

    # The answer always uses the chunk as the full context
    chunk = RetrievedChunk(text=text, source_id=source_id, source_name=source_name, score=1.0)
    context = build_context([chunk])

    # Build answer from the chunk text itself — for training we use the raw
    # chunk text as the "ideal" answer since it's ground truth from the docs
    answer_base = text.strip()
    if len(answer_base) > 400:
        answer_base = answer_base[:400] + "..."
    answer = f"{answer_base} [{source_id}]"

    if content_type == "price":
        items = _extract_items(text)
        for item in items or ["our products"]:
            q = random.choice(QUESTION_TEMPLATES["price"]).format(item=item)
            examples.append(format_training_example(business_name, context, q, answer))

    elif content_type == "hours":
        for template in random.sample(QUESTION_TEMPLATES["hours"], min(2, len(QUESTION_TEMPLATES["hours"]))):
            examples.append(format_training_example(business_name, context, template, answer))

    elif content_type == "policy":
        topics = _extract_topics(text)
        for topic in topics or ["this"]:
            q = random.choice(QUESTION_TEMPLATES["policy"]).format(topic=topic)
            examples.append(format_training_example(business_name, context, q, answer))

    else:
        items = _extract_items(text)
        for item in items or ["this"]:
            q = random.choice(QUESTION_TEMPLATES["item"]).format(item=item)
            examples.append(format_training_example(business_name, context, q, answer))

    # Always add one general question too
    general_q = random.choice(QUESTION_TEMPLATES["general"])
    examples.append(format_training_example(business_name, context, general_q, answer))

    return examples


def load_manual_qa(
    yaml_path: str,
    docs_dir: str,
    business_name: str,
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> list[dict[str, str]]:
    """
    Load manual Q&A pairs from eval_dataset.yaml (or any similar YAML).
    For each case that has an expected_answer, retrieve context from the
    index and create a proper training example.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    cases = data.get("cases", [])
    qa_cases = [c for c in cases if c.get("expected_answer") and c.get("expected_pass")]

    if not qa_cases:
        logger.info("No manual Q&A pairs found in %s", yaml_path)
        return []

    from inference.retriever import Retriever

    index_path = Path(docs_dir).parent / "faiss.index"
    metadata_path = Path(docs_dir).parent / "metadata.json"

    if not index_path.exists():
        logger.warning("No FAISS index found — skipping manual Q&A enrichment")
        return []

    retriever = Retriever(
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        embedding_model_name=embedding_model_name,
    )

    examples: list[dict[str, str]] = []
    for case in qa_cases:
        chunks = retriever.search(case["query"])
        if not chunks:
            continue
        context = build_context(chunks)
        examples.append(
            format_training_example(
                business_name,
                context,
                case["query"],
                case["expected_answer"],
            )
        )

    logger.info("Loaded %d manual Q&A examples from %s", len(examples), yaml_path)
    return examples


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate LoRA training data from documents")
    parser.add_argument("--docs-dir",      required=True)
    parser.add_argument("--output",        required=True, help="Path for .jsonl output")
    parser.add_argument("--business-name", required=True)
    parser.add_argument("--manual-qa",     default=None, help="Optional eval_dataset.yaml path")
    parser.add_argument("--chunk-size",    type=int, default=400)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    logger.info("Processing documents in %s ...", args.docs_dir)
    chunks = process_directory(
        args.docs_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    examples: list[dict[str, str]] = []

    for chunk in chunks:
        generated = generate_qa_from_chunk(
            text=chunk.text,
            source_id=chunk.source_id,
            source_name=chunk.source_name,
            business_name=args.business_name,
        )
        examples.extend(generated)

    if args.manual_qa:
        manual = load_manual_qa(args.manual_qa, args.docs_dir, args.business_name)
        examples.extend(manual)

    # Deduplicate and shuffle
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for ex in examples:
        key = ex["user"][:100]
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    random.shuffle(unique)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in unique:
            f.write(json.dumps(ex) + "\n")

    logger.info("Wrote %d training examples to %s", len(unique), out_path)

    # Print a sample
    if unique:
        sample = unique[0]
        logger.info("Sample example:")
        logger.info("  Q: %s", sample["user"][:100])
        logger.info("  A: %s", sample["assistant"][:100])


if __name__ == "__main__":
    main()
