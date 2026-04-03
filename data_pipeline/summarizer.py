"""Build synthetic summary chunks by calling the LLM at index time.

For each document we send its full text to the local Ollama model and ask
it to produce a structured natural-language summary (totals, top items,
categories).  The resulting chunk is added to the FAISS index so that
aggregation queries ("largest purchases", "total spending", "restaurant
charges") can be answered from a single retrieved chunk.

Because the LLM reads the raw text — not regex patterns — this works for
ANY document format: credit card statements, invoices, bakery receipts,
tax returns, sales reports, etc.  No hardcoded rules needed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import httpx

from data_pipeline.chunker import Chunk, _make_source_id, read_document

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"

_SUMMARY_PROMPT = """\
You are a financial document analyst. Read the document text below and \
produce a concise structured summary.

Include:
1. What kind of document this is and the time period it covers.
2. The total amount (new balance, invoice total, or equivalent) if present.
3. The top 10 largest individual line items or transactions with their \
amounts.
4. A breakdown of spending or charges grouped by category or vendor \
(infer categories from context — do NOT use hardcoded lists).
5. Any notable patterns (e.g. recurring charges, unusually large items).

Write in plain English. Be specific with numbers. Do not invent data that \
is not in the document.

Document ({source_name}):
---
{text}
---

Summary:"""


def _call_ollama_summary(text: str, source_name: str, cfg: dict) -> Optional[str]:
    """Ask Ollama to summarize a document. Returns None on failure."""
    model = cfg.get("model", {}).get("ollama_model", "phi3-finance")
    # Truncate to ~6000 chars so we stay within the model's context window
    truncated = text[:6000]
    if len(text) > 6000:
        truncated += "\n[... document truncated for summary ...]"

    prompt = _SUMMARY_PROMPT.format(source_name=source_name, text=truncated)

    try:
        resp = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 512},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning("Ollama summary failed for %s: %s", source_name, e)
        return None


def build_summary_chunk(
    path: Path,
    cfg: Optional[dict] = None,
) -> Chunk | None:
    """Read a document, call the LLM to summarise it, return a Chunk."""
    cfg = cfg or {}
    text = read_document(path)
    if not text.strip():
        return None

    logger.info("Generating LLM summary for %s ...", path.name)
    summary = _call_ollama_summary(text, path.name, cfg)

    if not summary:
        logger.warning("No summary generated for %s — skipping", path.name)
        return None

    # Wrap with a header so retrieval can match "summary" queries too
    full_text = (
        f"Document summary — {path.name}:\n"
        f"{summary}"
    )

    return Chunk(
        text=full_text,
        source_id=_make_source_id(path, 9999),
        source_name=path.name,
        chunk_index=9999,
    )


def build_summary_chunks(
    docs_dir: str | Path,
    cfg: Optional[dict] = None,
) -> list[Chunk]:
    """Return one LLM-generated summary Chunk per document in docs_dir."""
    cfg = cfg or {}
    docs_path = Path(docs_dir)
    summaries: list[Chunk] = []

    # Check Ollama is reachable before attempting
    try:
        httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
    except Exception:
        logger.warning(
            "Ollama not reachable at %s — skipping LLM summaries. "
            "Start Ollama and reindex to get summary chunks.",
            OLLAMA_BASE_URL,
        )
        return []

    for path in sorted(docs_path.iterdir()):
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".csv", ".docx"}:
            chunk = build_summary_chunk(path, cfg)
            if chunk:
                summaries.append(chunk)

    logger.info("Built %d LLM summary chunks from %s", len(summaries), docs_dir)
    return summaries
