# BabyYoday — Product Overview

## Purpose & Value Proposition

BabyYoday is a **private AI agent platform** that lets small business owners deploy a fully isolated, domain-restricted AI assistant powered exclusively by their own business data. No general-purpose chatbot answers, no hallucinations from the internet — just the business's own knowledge, running on their own hardware.

The core promise: upload your data → get a Dockerized agent that answers customer questions only from that data, refuses everything else, and stays current as new data is added.

## Key Features

- **Domain-restricted answering**: Three-layer enforcement ensures the agent never answers outside the business's data:
  1. Domain Gate — embedding similarity check against a precomputed centroid (rejects off-topic queries in ~5ms)
  2. RAG retrieval threshold — refuses if no relevant chunks found in FAISS
  3. System prompt grounding — model instructed to use only retrieved context

- **RAG pipeline**: Query → embed → FAISS vector search → top-K chunks → prompt assembly → SLM inference → validated response

- **Self-contained Docker container**: SLM weights, FAISS index, embedding model, domain centroid, and admin panel all baked into one image per customer

- **Live data updates**: File watcher monitors `/data/incoming/` and auto re-embeds + reindexes when new files are dropped in; admin panel provides browser-based upload

- **Multi-step agent reasoning**: Planner decomposes complex queries into sub-tasks; Executor runs each against FAISS + SLM; Validator checks citations and coherence

- **Response validation**: Citation verification, hallucination detection, provenance metadata attached to every response

- **Admin panel**: Web UI (FastAPI + Jinja2) for uploading new documents and viewing query logs

- **AWS cloud deployment**: Full CDK infrastructure — ECS Fargate, ALB, CloudFront, EFS, S3, ECR, CodePipeline

## Target Users & Use Cases

One container per customer. Each is completely independent — no shared database, no cross-customer data leakage.

| Business | Their data | Agent answers |
|----------|-----------|---------------|
| Bakery | Menu, prices, allergen info, hours, custom cake policies | "Do you have gluten-free cupcakes? What's the lead time for a wedding cake?" |
| Law firm | Practice areas, fee structures, intake procedures, FAQs | "Do you handle trademark disputes? What documents do I need?" |
| Travel agency | Packages, destinations, cancellation policies, visa requirements | "What's included in the Bali package? Can I cancel within 48 hours?" |
| SaaS product | Docs, pricing tiers, API guides, changelog | "How do I integrate the webhook? What's the rate limit on the Pro plan?" |
| Local gym | Class schedules, membership tiers, trainer bios, facility rules | "Is there a yoga class on Tuesdays? Can I freeze my membership?" |

## Delivery Model

1. Customer uploads their data (PDFs, DOCX, TXT, spreadsheets)
2. `build_customer.py` parses, chunks, embeds, builds FAISS index, computes domain centroid, bakes into Docker image
3. Container deployed to customer's own hardware (Mac Mini, AWS Lightsail, any VPS)
4. Customer gets: running agent API, API key, embeddable chat widget snippet, admin panel

## API Endpoints

- `POST /query` — RAG + SLM inference (inference service, port 8000)
- `GET /health` — liveness check (both services)
- `POST /upload` — add new documents (admin service, port 8001)
- `GET /` — admin dashboard (admin service, port 8001)
