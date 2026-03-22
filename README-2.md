# Domain-Locked SLM with RAG — Architecture & Implementation Plan

## Problem Statement

Build a system where a Small Language Model (SLM) answers questions **only** from our
domain data (e.g. travel or finance) — nothing else. The system must:

- Refuse to answer anything outside our domain.
- Stay current via an automated pipeline that ingests new data hourly/daily.
- Ship as a self-contained Docker container with a simple API.
- Support an agent layer for multi-step reasoning over domain data.

---

## Core Design Principle

> **Fine-tuning teaches the model your domain language.
> RAG restricts it to your domain knowledge.
> A domain gate refuses everything else.**

All three layers work together. RAG is the load-bearing wall — without it, a fine-tuned
model still happily answers general questions from its pretraining data.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT / USER                                  │
│                     (API call, chat UI, agent)                          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        API GATEWAY / AUTH                                │
│              (mTLS, API keys, rate limiting, RBAC)                       │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────┐       │
│   │              1. DOMAIN GATE (classifier)                     │       │
│   │                                                              │       │
│   │  • Embed the incoming query                                  │       │
│   │  • Compute cosine similarity against domain centroid         │       │
│   │  • If similarity < threshold → REJECT ("out of domain")     │       │
│   │  • Optionally: lightweight topic classifier (fastText/SVM)   │       │
│   └──────────────────────┬───────────────────────────────────────┘       │
│                          │ passes                                        │
│                          ▼                                               │
│   ┌──────────────────────────────────────────────────────────────┐       │
│   │              2. RETRIEVAL (RAG pipeline)                     │       │
│   │                                                              │       │
│   │  • Embed query using the same embedding model                │       │
│   │  • Search vector DB for top-K relevant chunks                │       │
│   │  • If no chunks above relevance threshold → REFUSE           │       │
│   │  • Assemble context window: retrieved chunks + metadata      │       │
│   └──────────────────────┬───────────────────────────────────────┘       │
│                          │ context                                       │
│                          ▼                                               │
│   ┌──────────────────────────────────────────────────────────────┐       │
│   │              3. PROMPT CONSTRUCTION                          │       │
│   │                                                              │       │
│   │  ┌────────────────────────────────────────────────────┐      │       │
│   │  │ SYSTEM: You are a {domain} assistant. Answer ONLY  │      │       │
│   │  │ using the provided context. If the context does    │      │       │
│   │  │ not contain the answer, say "I don't have          │      │       │
│   │  │ information on this." Cite source IDs.             │      │       │
│   │  ├────────────────────────────────────────────────────┤      │       │
│   │  │ CONTEXT: [retrieved chunks with source IDs]        │      │       │
│   │  ├────────────────────────────────────────────────────┤      │       │
│   │  │ USER: {original query}                             │      │       │
│   │  └────────────────────────────────────────────────────┘      │       │
│   └──────────────────────┬───────────────────────────────────────┘       │
│                          │ prompt                                        │
│                          ▼                                               │
│   ┌──────────────────────────────────────────────────────────────┐       │
│   │              4. SLM INFERENCE                                │       │
│   │                                                              │       │
│   │  • Small model (Mistral-7B / Phi-3 / Llama-3-8B)            │       │
│   │  • Optionally LoRA fine-tuned on domain Q&A pairs            │       │
│   │  • Generates answer grounded in retrieved context            │       │
│   │  • Returns: answer + citations + confidence                  │       │
│   └──────────────────────┬───────────────────────────────────────┘       │
│                          │ response                                      │
│                          ▼                                               │
│   ┌──────────────────────────────────────────────────────────────┐       │
│   │              5. RESPONSE VALIDATOR                           │       │
│   │                                                              │       │
│   │  • Verify citations map to real source IDs                   │       │
│   │  • Check for hallucination (answer vs. retrieved context)    │       │
│   │  • Strip any out-of-domain content that leaked through       │       │
│   │  • Attach provenance metadata to response                    │       │
│   └──────────────────────┬───────────────────────────────────────┘       │
│                          │                                               │
│             DOCKER CONTAINER (inference service)                         │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
                    Response to Client
```

---

## Data Pipeline (continuous updates)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                                        │
│   (APIs, databases, CSVs, feeds, support tickets, knowledge base)        │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   INGESTION PIPELINE                                     │
│                  (Airflow / Prefect / cron)                               │
│                                                                          │
│   ┌─────────────┐   ┌──────────────┐   ┌────────────────┐               │
│   │   Extract    │──▶│  Validate &  │──▶│  Chunk & Clean │               │
│   │   (fetch)    │   │  Deduplicate │   │  (text splits) │               │
│   └─────────────┘   └──────────────┘   └───────┬────────┘               │
│                                                 │                        │
└─────────────────────────────────────────────────┼───────────────────────┘
                                                  │
                      ┌───────────────────────────┼──────────────────┐
                      │                           │                  │
                      ▼                           ▼                  ▼
          ┌───────────────────┐     ┌──────────────────┐   ┌────────────────┐
          │   EMBEDDING       │     │  DOCUMENT STORE   │   │  TRAINING DATA │
          │   MODEL           │     │  (S3 / object     │   │  COLLECTOR     │
          │                   │     │   store)           │   │  (for future   │
          │  all-MiniLM-L6    │     │                    │   │   fine-tuning) │
          │  or domain-tuned  │     │  Raw + processed   │   │                │
          └────────┬──────────┘     │  documents with    │   │  Q&A pairs,    │
                   │                │  version history    │   │  feedback,     │
                   ▼                └──────────────────┘   │  corrections   │
          ┌───────────────────┐                             └────────────────┘
          │   VECTOR DB        │
          │                    │
          │  FAISS (local) or  │
          │  Milvus (prod)     │
          │                    │
          │  Upsert new        │
          │  embeddings on     │
          │  each pipeline run │
          └────────────────────┘
```

---

## Three Layers of Domain Enforcement

| Layer | Mechanism | What it catches | Speed |
|-------|-----------|----------------|-------|
| **1. Domain Gate** | Embedding similarity / topic classifier | Off-topic queries ("tell me a joke", "write python code") | ~5ms |
| **2. RAG Retrieval Threshold** | No relevant chunks found → refuse | Queries that sound domain-adjacent but have no matching data | ~50ms |
| **3. System Prompt + Grounding** | Model instructed to use only provided context | Prevents the model from using pretraining knowledge to fill gaps | Part of inference |

All three layers fire sequentially. A query must pass all three to get an answer.

---

## Component Details

### 1. Domain Gate

A lightweight, fast classifier that runs before anything expensive (retrieval, inference).

**Option A — Embedding centroid approach:**
- Precompute a centroid vector from all your domain documents.
- For each query, compute cosine similarity against the centroid.
- Reject if similarity < threshold (tune on a validation set).

**Option B — Trained classifier:**
- Train a small fastText or SVM classifier on (domain, not-domain) examples.
- Sub-millisecond inference, trivial to update.

### 2. RAG Pipeline

The core of the system. Every answer is grounded in retrieved documents.

- **Embedding model**: `all-MiniLM-L6-v2` (384-dim, fast, good quality) or a domain-fine-tuned variant.
- **Vector store**: FAISS for single-node / dev, Milvus or Weaviate for production.
- **Chunk strategy**: Split documents into 256–512 token chunks with ~50 token overlap. Attach metadata (source ID, timestamp, category).
- **Retrieval**: Top-K (K=5–10) chunks by cosine similarity. Apply a minimum relevance threshold — if the best chunk scores below it, refuse to answer.
- **Context assembly**: Concatenate retrieved chunks into a prompt context window, respecting the model's max token limit.

### 3. SLM (Inference)

The generative model that produces the final answer.

**Recommended base models (ranked by size):**

| Model | Parameters | RAM (quantized) | Best for |
|-------|-----------|-----------------|----------|
| Phi-3-mini | 3.8B | ~3 GB (4-bit) | Low-resource, fast responses |
| Mistral-7B | 7B | ~5 GB (4-bit) | Good balance of quality and speed |
| Llama-3-8B | 8B | ~6 GB (4-bit) | Strong reasoning, good instruction following |

**Optional LoRA fine-tuning:**
- Fine-tune with LoRA/QLoRA on domain-specific Q&A pairs.
- Improves answer quality and domain terminology, but is NOT the domain restriction mechanism (RAG does that).
- Keep the base model frozen; only train the adapter (~10–50 MB).
- Schedule periodic retrains as new labeled data accumulates.

### 4. Agent Layer

Orchestrates multi-step reasoning over the SLM + retrieval system.

```
                    ┌─────────────┐
                    │   PLANNER   │
                    │             │
                    │  Breaks the │
                    │  query into │
                    │  sub-tasks  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Retrieve │ │ Retrieve │ │ Compute  │
        │ context  │ │ related  │ │ derived  │
        │ for Q1   │ │ data Q2  │ │ answer   │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │             │            │
             └─────────────┼────────────┘
                           ▼
                    ┌─────────────┐
                    │  EXECUTOR   │
                    │             │
                    │  Calls SLM  │
                    │  with merged│
                    │  context    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  VALIDATOR  │
                    │             │
                    │  Checks     │
                    │  citations, │
                    │  coherence  │
                    └─────────────┘
```

- **Planner**: Decomposes complex queries into sub-tasks (can be rule-based or LLM-driven).
- **Executor**: Runs each sub-task against retrieval + SLM.
- **Validator**: Ensures the composed answer is grounded and consistent.
- Keep agent logic deterministic and fully logged for auditability.

### 5. Docker Container

The inference service is packaged as a single container.

```
┌──────────────────────────────────────────────┐
│              DOCKER CONTAINER                 │
│                                              │
│   ┌────────────────────────────────────┐     │
│   │         FastAPI Server             │     │
│   │                                    │     │
│   │  POST /query    → RAG + SLM       │     │
│   │  GET  /health   → liveness check  │     │
│   │  GET  /metrics  → Prometheus       │     │
│   └────────────────────────────────────┘     │
│                                              │
│   ┌──────────────┐  ┌───────────────────┐    │
│   │ SLM weights  │  │ Embedding model   │    │
│   │ + LoRA       │  │ (all-MiniLM)      │    │
│   │ adapter      │  │                   │    │
│   └──────────────┘  └───────────────────┘    │
│                                              │
│   ┌──────────────────────────────────────┐   │
│   │ FAISS index (loaded from volume/S3)  │   │
│   └──────────────────────────────────────┘   │
│                                              │
│   • Distroless / slim base image             │
│   • Non-root user                            │
│   • No network egress (firewall rules)       │
│   • No package manager in final image        │
└──────────────────────────────────────────────┘
```

---

## Request Flow (end to end)

```
User Query: "What are the cancellation fees for Tokyo flights in March?"

  1. API Gateway     → Authenticate, rate-limit
  2. Domain Gate     → Embed query, similarity=0.82 (threshold=0.6) → PASS
  3. RAG Retrieval   → Top 5 chunks from vector DB:
                        - [DOC-4521] Tokyo cancellation policy v3
                        - [DOC-4519] March seasonal pricing rules
                        - [DOC-3877] General cancellation fee structure
                        - [DOC-4102] Tokyo route information
                        - [DOC-3901] Refund processing timelines
  4. Prompt Build    → System prompt + 5 chunks + user query
  5. SLM Inference   → "Cancellation fees for Tokyo flights in March are
                        ¥5,000 for bookings cancelled 7+ days before departure
                        and ¥15,000 within 7 days. [DOC-4521, DOC-4519]"
  6. Validator       → Citations DOC-4521, DOC-4519 exist ✓, answer in context ✓
  7. Return          → JSON response with answer + sources + confidence

---

User Query: "What's the recipe for chocolate cake?"

  1. API Gateway     → Authenticate, rate-limit
  2. Domain Gate     → Embed query, similarity=0.12 (threshold=0.6) → REJECT
  3. Return          → {"error": "This question is outside our domain."}
```

---

## Technology Stack

| Component | Tool | Role |
|-----------|------|------|
| **Embedding model** | `all-MiniLM-L6-v2` or domain-tuned | Encode queries + documents into vectors |
| **Vector database** | FAISS (dev/single-node), Milvus (production) | Store and search document embeddings |
| **SLM base model** | Mistral-7B / Phi-3-mini / Llama-3-8B | Generate answers from retrieved context |
| **Fine-tuning** | HuggingFace Transformers + PEFT (LoRA) | Optional domain adaptation of the SLM |
| **Inference server** | FastAPI + Uvicorn | Serve the SLM behind a REST API |
| **Quantization** | bitsandbytes / GPTQ / GGUF | Reduce model size for CPU/low-GPU deployment |
| **Orchestration** | Airflow / Prefect | Schedule data ingestion + embedding pipelines |
| **Container** | Docker (distroless base) | Package and ship the inference service |
| **Orchestration (prod)** | Kubernetes + Helm | Scale and manage containers in production |
| **Monitoring** | Prometheus + Grafana | Track latency, throughput, error rates |
| **CI/CD** | GitHub Actions | Build images, run tests, deploy |
| **Security** | Vault (secrets), OPA (policies) | Manage credentials and enforce access rules |

---

## Repository Layout

```
babyyoday/
├── data_ingest/              # ETL scripts, Airflow DAGs
│   ├── sources/              # Per-source extractors
│   ├── validators/           # Schema checks, dedup
│   └── dags/                 # Pipeline definitions
│
├── embeddings/               # Embedding generation
│   ├── embed.py              # Batch embedding script
│   ├── chunker.py            # Document chunking logic
│   └── upsert.py             # Vector DB upsert
│
├── retrieval/                # RAG pipeline (core)
│   ├── retriever.py          # Query → vector search → top-K
│   ├── context_builder.py    # Assemble prompt context
│   └── domain_gate.py        # Out-of-domain classifier
│
├── model_training/           # Optional LoRA fine-tuning
│   ├── train_lora.py         # Training script
│   ├── eval.py               # Holdout evaluation
│   └── configs/              # Hyperparameter configs
│
├── inference/                # Inference service (Docker)
│   ├── server.py             # FastAPI application
│   ├── prompt.py             # Prompt template construction
│   ├── validator.py          # Response validation
│   ├── Dockerfile            # Container definition
│   └── requirements.txt      # Python dependencies
│
├── agent/                    # Agent orchestration
│   ├── planner.py            # Task decomposition
│   ├── executor.py           # Sub-task execution
│   └── router.py             # Query routing logic
│
├── ci/                       # CI/CD pipelines
│   └── .github/workflows/    # GitHub Actions
│
├── infra/                    # Infrastructure
│   ├── k8s/                  # Kubernetes manifests
│   ├── helm/                 # Helm chart
│   └── docker-compose.yml    # Local dev environment
│
├── tests/                    # Test suite
│   ├── test_domain_gate.py   # Domain classifier tests
│   ├── test_retrieval.py     # RAG pipeline tests
│   └── test_inference.py     # End-to-end inference tests
│
├── Readme                    # Original brainstorm
└── README-2.md               # This document
```

---

## Implementation Roadmap

### Phase 1 — Foundation (Weeks 1–4)

| Week | Deliverable |
|------|------------|
| 1 | Data inventory. Identify sources, define schemas, build first extractor. |
| 2 | Chunking + embedding pipeline. Embed seed documents into FAISS. |
| 3 | RAG pipeline: query embedding → vector search → context assembly. |
| 4 | Domain gate: embedding centroid classifier, tune threshold on test queries. |

**Milestone**: You can ask a domain question and get retrieved context back. Off-topic queries are rejected.

### Phase 2 — Inference (Weeks 5–8)

| Week | Deliverable |
|------|------------|
| 5 | Load a quantized SLM (Mistral-7B-4bit). Wire up prompt construction. |
| 6 | FastAPI server with `/query` endpoint. End-to-end RAG + SLM flow works. |
| 7 | Response validator (citation checking, grounding verification). |
| 8 | Dockerize. Hardened container, health checks, Prometheus metrics. |

**Milestone**: A Docker container that accepts queries and returns grounded, domain-only answers.

### Phase 3 — Agent & Pipeline (Weeks 9–12)

| Week | Deliverable |
|------|------------|
| 9 | Agent planner + executor for multi-step queries. |
| 10 | Automated data ingestion pipeline (hourly/daily). Embedding upsert on new data. |
| 11 | CI/CD: image builds, automated tests, deployment pipeline. |
| 12 | Monitoring dashboard. Logging and audit trail. |

**Milestone**: Fully automated system that updates itself with new data and serves queries through an agent.

### Phase 4 — Hardening (Weeks 13–16)

| Week | Deliverable |
|------|------------|
| 13 | Optional LoRA fine-tuning on accumulated domain Q&A pairs. |
| 14 | Canary deployment: shadow mode testing of fine-tuned model. |
| 15 | Feedback loop: user ratings → training data collector. |
| 16 | Security audit, PII handling, compliance checks, production sign-off. |

**Milestone**: Production-ready, continuously improving system.

---

## Key Decisions to Make Before Starting

1. **Which domain?** Travel or finance — this determines data sources and compliance requirements.
2. **Base model size?** 3.8B (Phi-3, fast, low resource) vs 7–8B (better quality, needs more GPU/RAM).
3. **Deployment target?** Single machine with GPU, or Kubernetes cluster?
4. **Vector DB?** FAISS is simpler for dev/single-node; Milvus/Weaviate for multi-node production.
5. **Fine-tune or RAG-only first?** Start RAG-only. Add LoRA fine-tuning later once you have labeled Q&A data.
