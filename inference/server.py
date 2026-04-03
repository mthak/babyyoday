from __future__ import annotations

import logging
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

from data_pipeline.watcher import start_watcher_nonblocking
from inference.context_builder import build_context, extract_source_ids
from inference.domain_gate import DomainGate
from inference.prompt import build_system_prompt
from inference.retriever import Retriever
from inference.validator import validate_response

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("/app/config.yaml")
LOCAL_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

OLLAMA_BASE_URL = "http://localhost:11434"


def load_config() -> dict:
    path = CONFIG_PATH if CONFIG_PATH.exists() else LOCAL_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def _check_ollama(model_name: str) -> bool:
    """Return True if Ollama is running and the model is available."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        available = any(model_name in m for m in models)
        if available:
            logger.info("Ollama ready — model '%s' found", model_name)
        else:
            logger.warning("Ollama running but model '%s' not found. Available: %s", model_name, models)
        return available
    except Exception as e:
        logger.warning("Ollama not reachable: %s", e)
        return False


def _call_ollama(cfg: dict, context: str, query: str) -> str:
    """Call Ollama's chat API with the assembled context."""
    business_name = cfg["business_name"]
    model_name = cfg["model"].get("ollama_model", "phi3-finance")
    temperature = cfg["model"].get("temperature", 0.3)
    max_tokens = cfg["model"].get("max_tokens", 256)

    system_prompt = build_system_prompt(business_name)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "stream": False,
    }

    resp = httpx.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    _state["config"] = cfg

    retriever = Retriever(
        index_path=cfg["faiss"]["index_path"],
        metadata_path=cfg["faiss"]["metadata_path"],
        embedding_model_name=cfg["embedding"]["model_name"],
        top_k=cfg["retrieval"]["top_k"],
        relevance_threshold=cfg["retrieval"]["relevance_threshold"],
    )
    _state["retriever"] = retriever

    _state["domain_gate"] = DomainGate(
        centroid_path=cfg["domain_gate"]["centroid_path"],
        similarity_threshold=cfg["domain_gate"]["similarity_threshold"],
    )

    ollama_model = cfg["model"].get("ollama_model", "phi3-finance")
    _state["use_ollama"] = _check_ollama(ollama_model)
    if not _state["use_ollama"]:
        logger.warning("Ollama not available — running in retrieval-only mode")

    observer = start_watcher_nonblocking(
        watch_dir=cfg["data"]["watch_dir"],
        docs_dir=cfg["data"]["docs_dir"],
        output_dir=str(Path(cfg["faiss"]["index_path"]).parent),
        retriever=retriever,
        config=cfg,
    )
    _state["watcher"] = observer

    logger.info("Server ready — business: %s", cfg["business_name"])
    yield

    observer.stop()
    observer.join()
    _state.clear()


app = FastAPI(title="BabyYoday Agent", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    domain_score: float
    latency_ms: float
    grounded: bool
    mode: str  # "llm" or "retrieval-only"


class ErrorResponse(BaseModel):
    error: str
    domain_score: Optional[float] = None


@app.get("/health")
async def health():
    cfg = _state.get("config", {})
    return {
        "status": "ok",
        "business": cfg.get("business_name", "unknown"),
        "ollama_ready": _state.get("use_ollama", False),
        "ollama_model": cfg.get("model", {}).get("ollama_model", "phi3-finance"),
        "index_size": _state["retriever"].index.ntotal if "retriever" in _state else 0,
    }


@app.post("/query")
async def query(req: QueryRequest):
    t0 = time.time()
    cfg = _state["config"]
    retriever: Retriever = _state["retriever"]
    gate: DomainGate = _state["domain_gate"]

    logger.info("─── NEW QUERY: %r", req.query)

    logger.info("  [1/5] Embedding query...")
    query_embedding = retriever.embed_query(req.query)
    logger.info("  [1/5] Done. Checking domain gate...")
    allowed, similarity = gate.check(query_embedding)
    logger.info("  [1/5] Domain gate: score=%.4f  allowed=%s", similarity, allowed)

    if not allowed:
        return ErrorResponse(
            error=(
                f"I can only help with questions about {cfg['business_name']}. "
                "How can I help with that?"
            ),
            domain_score=similarity,
        )

    logger.info("  [2/5] Searching FAISS index (top_k=%d, threshold=%.2f)...",
                cfg["retrieval"]["top_k"], cfg["retrieval"]["relevance_threshold"])
    chunks = retriever.search(req.query)
    logger.info("  [2/5] Retrieved %d chunks", len(chunks))
    for i, c in enumerate(chunks):
        logger.info("        chunk[%d] score=%.4f  src=%s", i, c.score, c.source_name)

    if not chunks:
        return ErrorResponse(
            error="I don't have information on that topic.",
            domain_score=similarity,
        )

    n_ctx = cfg["model"].get("n_ctx", 4096)
    max_gen = cfg["model"].get("max_tokens", 256)
    context_budget = n_ctx - max_gen - 350

    logger.info("  [3/5] Building context (budget=%d tokens)...", context_budget)
    context = build_context(chunks, max_tokens=context_budget)
    source_ids = extract_source_ids(chunks)
    logger.info("  [3/5] Context built: %d chars, %d sources", len(context), len(source_ids))

    if not _state.get("use_ollama"):
        answer_text = (
            f"[Retrieval-only mode] Found {len(chunks)} relevant chunk(s). "
            f"Sources: {', '.join(source_ids)}. "
            "Start Ollama and run: ollama create phi3-finance -f Modelfile"
        )
        mode = "retrieval-only"
    else:
        logger.info("  [4/5] Calling Ollama (max_tokens=%d)...", max_gen)
        answer_text = _call_ollama(cfg, context, req.query)
        logger.info("  [4/5] Ollama done. Answer length: %d chars", len(answer_text))
        mode = "llm"

    logger.info("  [5/5] Validating response...")
    validation = validate_response(answer_text, source_ids)
    latency = (time.time() - t0) * 1000
    logger.info("  [5/5] Done. Total latency: %.0fms", latency)

    return QueryResponse(
        answer=validation.answer,
        sources=[
            {"id": c.source_id, "name": c.source_name, "score": round(c.score, 3)}
            for c in chunks
        ],
        domain_score=round(similarity, 3),
        latency_ms=round(latency, 1),
        grounded=validation.is_valid,
        mode=mode,
    )
