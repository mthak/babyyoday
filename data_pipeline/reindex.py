from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding

from data_pipeline.chunker import Chunk, process_directory
from data_pipeline.summarizer import build_summary_chunks

logger = logging.getLogger(__name__)


def build_index(
    chunks: list[Chunk],
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple[faiss.IndexFlatIP, list[dict], np.ndarray]:
    """Embed chunks and build a FAISS inner-product index.

    Returns (index, metadata_list, embeddings_matrix).
    """
    model = TextEmbedding(embedding_model_name)
    texts = [c.text for c in chunks]

    logger.info("Embedding %d chunks with %s (ONNX) ...", len(texts), embedding_model_name)
    embeddings = np.array(list(model.embed(texts)), dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    metadata = [
        {
            "text": c.text,
            "source_id": c.source_id,
            "source_name": c.source_name,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]

    logger.info("FAISS index built — %d vectors, dim=%d", index.ntotal, dim)
    return index, metadata, embeddings


def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    centroid = embeddings.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
    return centroid


def save_index(
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    centroid: np.ndarray,
    output_dir: str | Path,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out / "faiss.index"))

    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    np.save(str(out / "centroid.npy"), centroid)

    logger.info("Index saved to %s", out)


def reindex(
    docs_dir: str,
    output_dir: str,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    cfg: dict | None = None,
):
    """Full reindex: read docs → chunk + summaries → embed → save."""
    chunks = process_directory(docs_dir, chunk_size, chunk_overlap)
    summary_chunks = build_summary_chunks(docs_dir, cfg=cfg)

    if summary_chunks:
        logger.info("Adding %d summary chunks to index", len(summary_chunks))
        chunks = chunks + summary_chunks

    if not chunks:
        logger.warning("No chunks produced from %s — skipping index build", docs_dir)
        return

    index, metadata, embeddings = build_index(chunks, embedding_model_name)
    centroid = compute_centroid(embeddings)
    save_index(index, metadata, centroid, output_dir)


if __name__ == "__main__":
    import argparse

    import yaml

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Reindex documents into FAISS")
    parser.add_argument("--docs-dir", required=True, help="Directory with documents")
    parser.add_argument("--output-dir", required=True, help="Where to save the index")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = {}
    try:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logging.warning("Config file %s not found — using defaults", args.config)

    reindex(args.docs_dir, args.output_dir, args.model, args.chunk_size, args.chunk_overlap, cfg=cfg)
