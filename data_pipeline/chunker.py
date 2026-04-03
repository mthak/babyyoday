from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".csv"}


@dataclass
class Chunk:
    text: str
    source_id: str
    source_name: str
    chunk_index: int


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        logger.warning("pypdf not installed — skipping %s", path.name)
        return ""


def _read_csv(path: Path) -> str:
    """Convert CSV rows into human-readable sentences for embedding.

    Each row becomes a sentence like:
      "name: Chocolate Cake | price: $32 | category: vegan"
    This makes the content semantically searchable.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            return _read_text_file(path)

        lines: list[str] = []
        for row in rows:
            parts = [f"{k.strip()}: {v.strip()}" for k, v in row.items() if v and v.strip()]
            if parts:
                lines.append(" | ".join(parts))

        result = "\n".join(lines)
        logger.info("CSV %s: %d rows converted to text", path.name, len(rows))
        return result
    except Exception as e:
        logger.warning("CSV parse failed for %s (%s) — falling back to raw text", path.name, e)
        return _read_text_file(path)


def _read_docx(path: Path) -> str:
    try:
        from docx import Document

        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        logger.warning("python-docx not installed — skipping %s", path.name)
        return ""


def read_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return _read_text_file(path)
    elif ext == ".csv":
        return _read_csv(path)
    elif ext == ".pdf":
        return _read_pdf(path)
    elif ext == ".docx":
        return _read_docx(path)
    else:
        logger.warning("Unsupported file type: %s", ext)
        return ""


def _make_source_id(path: Path, chunk_idx: int) -> str:
    name_hash = hashlib.md5(path.name.encode()).hexdigest()[:6].upper()
    return f"DOC-{name_hash}-{chunk_idx}"


def chunk_text(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split text into overlapping chunks by character count."""
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - chunk_overlap

    return chunks


# Matches lines like: "12/06  AMAZON MKTPL*ABC  Amzn.com/bill WA  8.81"
_TRANSACTION_LINE = re.compile(
    r"^\d{2}/\d{2}\s+\S+.*\d+\.\d{2}\s*$", re.MULTILINE
)


def _enrich_chunk(text: str, source_name: str) -> str:
    """Prepend a semantic label to chunks that contain transaction data.

    This bridges the vocabulary gap between natural-language queries
    ('largest purchases', 'restaurant spending') and raw statement lines
    ('12/06 GRUBHUB 79.53') so the embedding model can match them.
    """
    transaction_lines = _TRANSACTION_LINE.findall(text)
    if len(transaction_lines) >= 2:
        return f"Credit card transactions — purchases, charges, and payments:\n{text}"
    return text


def process_file(
    path: Path,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    text = read_document(path)
    if not text.strip():
        return []

    raw_chunks = chunk_text(text, chunk_size, chunk_overlap)
    result: list[Chunk] = []
    for i, text_chunk in enumerate(raw_chunks):
        enriched = _enrich_chunk(text_chunk, path.name)
        result.append(
            Chunk(
                text=enriched,
                source_id=_make_source_id(path, i),
                source_name=path.name,
                chunk_index=i,
            )
        )

    logger.info("Chunked %s → %d chunks", path.name, len(result))
    return result


def process_directory(
    docs_dir: str | Path,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    docs_path = Path(docs_dir)
    all_chunks: list[Chunk] = []

    for path in sorted(docs_path.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            all_chunks.extend(process_file(path, chunk_size, chunk_overlap))

    logger.info("Processed %s → %d total chunks", docs_dir, len(all_chunks))
    return all_chunks
