from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI
from pydantic import BaseModel

from data_pipeline.watcher import start_watcher_nonblocking
from inference.context_builder import build_context, extract_source_ids
from inference.domain_gate import DomainGate
from inference.prompt import CHAT_FORMATS, build_chat_messages, format_for_completion
from inference.retriever import Retriever
from inference.validator import validate_response

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("/app/config.yaml")
LOCAL_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config() -> dict:
    path = CONFIG_PATH if CONFIG_PATH.exists() else LOCAL_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def _detect_chat_format(model_path: str, cfg: dict) -> Optional[str]:
    """Detect chat template from model filename or explicit config."""
    explicit = cfg.get("model", {}).get("chat_format")
    if explicit:
        return explicit

    name = Path(model_path).name.lower()
    if "phi" in name:
        return CHAT_FORMATS["phi3"]
    if "mistral" in name or "mixtral" in name:
        return CHAT_FORMATS["mistral"]
    if "llama-3" in name or "llama3" in name:
        return CHAT_FORMATS["llama3"]
    # Safe default for most instruction-tuned models
    return CHAT_FORMATS["chatml"]


def _load_llm(cfg: dict):
    """Load llama_cpp model if the file exists."""
    model_path = cfg["model"]["path"]
    if not Path(model_path).exists():
        logger.warning("Model not found at %s — running in retrieval-only mode", model_path)
        return None

    try:
        from llama_cpp import Llama
    except ImportError:
        logger.warning("llama-cpp-python not installed — retrieval-only mode")
        return None

    chat_format = _detect_chat_format(model_path, cfg)
    logger.info("Loading LLM from %s (chat_format=%s)", model_path, chat_format)

    llm = Llama(
        model_path=model_path,
        n_ctx=cfg["model"].get("n_ctx", 2048),
        n_gpu_layers=cfg["model"].get("n_gpu_layers", 0),
        chat_format=chat_format,
        verbose=False,
    )
    logger.info("LLM ready")
    return llm


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

    _state["llm"] = _load_llm(cfg)

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


def _call_llm(llm, cfg: dict, context: str, query: str) -> str:
    """Call the LLM using chat completion (preferred) with plain completion fallback."""
    business_name = cfg["business_name"]
    temperature = cfg["model"].get("temperature", 0.3)
    max_tokens = cfg["model"].get("max_tokens", 512)

    messages = build_chat_messages(business_name, context, query)

    try:
        result = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["<|end|>", "<|eot_id|>", "[/INST]"],
        )
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning("Chat completion failed (%s), falling back to completion", e)
        prompt = format_for_completion(business_name, context, query)
        result = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\n\nQuestion:", "\n\n---", "<|end|>"],
        )
        return result["choices"][0]["text"].strip()


@app.get("/health")
def health():
    cfg = _state.get("config", {})
    return {
        "status": "ok",
        "business": cfg.get("business_name", "unknown"),
        "model_loaded": _state.get("llm") is not None,
        "model_path": cfg.get("model", {}).get("path", ""),
        "index_size": _state["retriever"].index.ntotal if "retriever" in _state else 0,
    }


@app.post("/query")
def query(req: QueryRequest):
    t0 = time.time()
    cfg = _state["config"]
    retriever: Retriever = _state["retriever"]
    gate: DomainGate = _state["domain_gate"]

    query_embedding = retriever.embed_query(req.query)
    allowed, similarity = gate.check(query_embedding)

    if not allowed:
        return ErrorResponse(
            error=(
                f"I can only help with questions about {cfg['business_name']}. "
                "How can I help with that?"
            ),
            domain_score=similarity,
        )

    chunks = retriever.search(req.query)

    if not chunks:
        return ErrorResponse(
            error="I don't have information on that topic.",
            domain_score=similarity,
        )

    context = build_context(chunks)
    source_ids = extract_source_ids(chunks)

    llm = _state.get("llm")
    if llm is None:
        answer_text = (
            f"[Retrieval-only mode] Found {len(chunks)} relevant chunk(s). "
            f"Sources: {', '.join(source_ids)}. "
            "Add a model.gguf file to get full natural-language answers."
        )
        mode = "retrieval-only"
    else:
        answer_text = _call_llm(llm, cfg, context, req.query)
        mode = "llm"

    validation = validate_response(answer_text, source_ids)
    latency = (time.time() - t0) * 1000

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
