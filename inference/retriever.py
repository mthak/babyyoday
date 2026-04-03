from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source_id: str
    source_name: str
    score: float


class Retriever:
    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5,
        relevance_threshold: float = 0.3,
    ):
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold

        logger.info("Loading embedding model (ONNX/CPU): %s", embedding_model_name)
        self.embedding_model = TextEmbedding(embedding_model_name)

        logger.info("Loading FAISS index from: %s", index_path)
        self.index = faiss.read_index(index_path)

        logger.info("Loading metadata from: %s", metadata_path)
        with open(metadata_path) as f:
            self.metadata: list[dict] = json.load(f)

        logger.info(
            "Retriever ready — %d chunks indexed", self.index.ntotal
        )

    def reload_index(self, index_path: str, metadata_path: str):
        """Hot-reload index after new data is ingested."""
        self.index = faiss.read_index(index_path)
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        logger.info("Index reloaded — %d chunks", self.index.ntotal)

    def embed_query(self, query: str) -> np.ndarray:
        embeddings = list(self.embedding_model.embed([query]))
        return np.array(embeddings, dtype=np.float32)

    def search(self, query: str) -> list[RetrievedChunk]:
        query_vector = self.embed_query(query)
        scores, indices = self.index.search(query_vector, self.top_k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < self.relevance_threshold:
                continue
            meta = self.metadata[idx]
            results.append(
                RetrievedChunk(
                    text=meta["text"],
                    source_id=meta["source_id"],
                    source_name=meta.get("source_name", "unknown"),
                    score=float(score),
                )
            )

        logger.info(
            "Query: %s — %d chunks above threshold %.2f",
            query[:80],
            len(results),
            self.relevance_threshold,
        )
        return results
