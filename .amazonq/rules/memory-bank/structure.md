# BabyYoday — Project Structure

## Directory Layout

```
babyyoday/
├── builder/                  # Build tooling to produce a customer's Docker image
│   ├── build_customer.py     # Main entry: customer data dir → Docker image
│   ├── build_gate.py         # Compute domain centroid from embeddings
│   ├── embed_data.py         # Chunk + embed customer documents into FAISS
│   ├── config_template.yaml  # Per-customer config template
│   ├── Dockerfile            # Container definition (python:3.11-slim base)
│   └── entrypoint.sh         # Container startup: model download, index seed, uvicorn
│
├── inference/                # Runs inside the customer's container (port 8000)
│   ├── server.py             # FastAPI application — /query, /health endpoints
│   ├── domain_gate.py        # Embedding similarity check against centroid
│   ├── retriever.py          # Query → FAISS search → top-K chunks
│   ├── context_builder.py    # Assemble prompt context from retrieved chunks
│   ├── prompt.py             # Prompt template construction
│   ├── validator.py          # Response validation: citations, grounding
│   └── requirements.txt      # Python dependencies
│
├── agent/                    # Multi-step reasoning orchestration (inside container)
│   ├── planner.py            # Decompose complex queries into sub-tasks
│   ├── executor.py           # Run sub-tasks against FAISS + SLM
│   └── router.py             # Query routing logic
│
├── admin/                    # Admin panel (inside container, port 8001)
│   ├── app.py                # FastAPI app — upload docs, view query logs
│   └── templates/
│       └── dashboard.html    # Jinja2 dashboard template
│
├── data_pipeline/            # Runs inside container on schedule/watch
│   ├── watcher.py            # Watchdog: monitor /data/incoming/ for new files
│   ├── chunker.py            # Document chunking logic (PDF, DOCX, TXT, XLSX)
│   └── reindex.py            # Re-embed and rebuild FAISS index + centroid
│
├── infra/                    # AWS CDK infrastructure (TypeScript)
│   ├── bin/app.ts            # CDK app entry point — instantiates all stacks
│   └── lib/
│       ├── network-stack.ts  # VPC, subnets, NAT gateway, security groups
│       ├── storage-stack.ts  # S3 bucket (docs/models), EFS filesystem
│       ├── ecr-stack.ts      # ECR repository
│       ├── ecs-stack.ts      # Fargate cluster, task defs, services, ALB, Secrets Manager
│       ├── cdn-stack.ts      # CloudFront distribution
│       └── pipeline-stack.ts # CodePipeline + CodeBuild (GitHub → ECR → ECS)
│
├── data/
│   ├── docs/                 # Business documents (allergens.txt, menu.txt, etc.)
│   └── incoming/             # Drop zone for new files (watched by watcher.py)
│
├── sample_data/              # Sample bakery data for local dev/testing
├── models/                   # GGUF model weights (mounted from EFS at runtime)
├── tests/                    # pytest test suite
├── config.yaml               # Runtime config (model path, embedding, retrieval, etc.)
├── docker-compose.yml        # Local dev: spin up the full agent
├── requirements.txt          # Top-level Python dependencies
└── setup_local.py            # Local environment setup script
```

## Core Components & Relationships

```
Customer Query
     │
     ▼
inference/server.py (FastAPI :8000)
     │
     ├─► inference/domain_gate.py   ← centroid.npy (precomputed at build time)
     │         │ REJECT if similarity < threshold
     │         │ PASS
     ▼
     ├─► inference/retriever.py     ← faiss.index + metadata.json
     │         │ top-K chunks
     ▼
     ├─► inference/context_builder.py
     │         │ assembled context
     ▼
     ├─► inference/prompt.py
     │         │ formatted prompt
     ▼
     ├─► llama-cpp-python (SLM inference)
     │         │ raw answer
     ▼
     └─► inference/validator.py     → validated response + citations
```

```
New Document Upload
     │
     ├─► admin/app.py (FastAPI :8001) — browser upload
     │         │
     └─► data_pipeline/watcher.py   — file drop in /data/incoming/
               │
               ▼
         data_pipeline/chunker.py   — parse + chunk (400 tokens, 50 overlap)
               │
               ▼
         data_pipeline/reindex.py   — embed + rebuild FAISS + recompute centroid
```

```
Build Time (builder/)
     │
     ├─► build_customer.py          — orchestrates full build
     │         │
     ├─► embed_data.py              — chunk + embed all customer docs
     │         │
     └─► build_gate.py              — compute domain centroid
               │
               ▼
         Docker image (FAISS index + centroid + config baked in)
```

## AWS Architecture

```
Internet → CloudFront (HTTPS) → ALB (HTTP) → ECS Fargate
                                               ├── InferenceService :8000  /query /health
                                               └── AdminService     :8001  /  /upload
                                                       │
                                                      EFS (FAISS index, docs, query logs)
                                                      S3  (document uploads, model weights)
                                                      ECR (Docker image)
                                                      Secrets Manager (API key)
```

CDK Stack deployment order:
1. `BabyYodayNetwork` — VPC + networking
2. `BabyYodayStorage` — S3 + EFS
3. `BabyYodayEcr` — ECR repository
4. `BabyYodayEcs` — Fargate cluster + ALB
5. `BabyYodayCdn` — CloudFront
6. `BabyYodayPipeline` — CI/CD pipeline

## Architectural Patterns

- **One container per customer**: Fully isolated deployments, no shared infrastructure
- **Build-time vs runtime data**: FAISS index and centroid baked at build time; live updates via watcher/admin at runtime
- **Sequential gate pattern**: Domain Gate → RAG threshold → prompt grounding — all three must pass
- **Shared image, split services**: Same Docker image runs both InferenceService and AdminService; AdminService overrides entrypoint to skip ML model loading
- **EFS for persistence**: FAISS index, docs, and query logs stored on EFS so they survive container restarts and are shared between task revisions
- **Separate artifact per service**: CodeBuild produces `imagedefinitions-inference.json` and `imagedefinitions-admin.json` so each EcsDeployAction deploys independently
