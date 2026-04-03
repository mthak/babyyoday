from __future__ import annotations

from typing import Callable, Optional

from inference.retriever import RetrievedChunk


def build_context(
    chunks: list[RetrievedChunk],
    max_tokens: int = 1500,
    tokenize: Optional[Callable[[str], list]] = None,
) -> str:
    """Assemble retrieved chunks into a context string.

    Uses the model's actual tokenizer when available so the budget is exact.
    Falls back to a conservative 1 char ≈ 1 token estimate (safe for
    financial/numeric text) when running in retrieval-only mode.
    """
    parts: list[str] = []
    used_tokens = 0

    def count(text: str) -> int:
        if tokenize is not None:
            return len(tokenize(text.encode()))
        return len(text)  # 1 char ≈ 1 token fallback

    for chunk in chunks:
        entry = f"[{chunk.source_id}] ({chunk.source_name})\n{chunk.text}"
        entry_tokens = count(entry)
        if used_tokens + entry_tokens > max_tokens:
            break
        parts.append(entry)
        used_tokens += entry_tokens

    return "\n\n---\n\n".join(parts)


def extract_source_ids(chunks: list[RetrievedChunk]) -> list[str]:
    return [c.source_id for c in chunks]
