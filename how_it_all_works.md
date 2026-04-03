# How It All Works

A local RAG (Retrieval-Augmented Generation) system for querying personal
financial documents — or any document collection — using a fully offline
LLM stack.

---

## The Big Picture

```
                        ┌─────────────────────────────────────┐
                        │           INDEX TIME (once)          │
                        │                                      │
  PDFs / docs ──────────┤  1. chunker   → raw text chunks      │
                        │  2. summarizer (Ollama) → 1 summary  │
                        │     chunk per doc                    │
                        │  3. fastembed → vectors              │
                        │  4. FAISS index saved to disk        │
                        └─────────────────────────────────────┘

                        ┌─────────────────────────────────────┐
                        │         QUERY TIME (every request)   │
                        │                                      │
  User question ────────┤  1. fastembed → query vector        │
                        │  2. FAISS search → top-k chunks     │
                        │  3. Ollama LLM → natural language    │
                        │     answer                           │
                        └─────────────────────────────────────┘
```

---

## Step-by-Step: Index Time

This runs once (or whenever you add new documents) via:

```bash
python -m data_pipeline.reindex \
  --docs-dir data/docs/credit_cards \
  --output-dir data/output
```

### 1. Read and Chunk (`data_pipeline/chunker.py`)

Each document is read and split into overlapping chunks of ~400 characters
(`chunking.chunk_size` in `config.yaml`).

- PDF files are read with `pypdf`
- Chunks overlap by 50 characters so no sentence is cut off mid-thought
- Transaction-pattern chunks (lines with `MM/DD MERCHANT AMOUNT`) get a
  semantic prefix prepended:
  `"Credit card transactions — purchases, charges, and payments:"`
  This bridges the vocabulary gap between raw statement lines and
  natural-language queries.

### 2. LLM Summary (`data_pipeline/summarizer.py`)

For every document, the full text is sent to the **local Ollama LLM** with
a prompt asking it to:
- Identify the document type and time period
- Extract the total amount
- List the top 10 largest line items
- Group charges by category (inferred from context — no hardcoded lists)
- Note any patterns

The LLM returns a rich natural-language summary like:

```
Document summary — 20250105-statements-3544-.pdf:
This is an Amazon Chase credit card statement for December 2024 –
January 2025. Total new balance: $1,148.65 across 47 transactions.

Top transactions by amount:
  12/12  Amazon Marketplace     $88.18
  01/04  Grubhub                $79.53
  12/15  Whole Foods            $78.06
  ...

Spending by category:
  Amazon purchases:  $623.40
  Whole Foods:        $90.59
  Food delivery:      $79.53
  Other:             $355.13
```

This summary becomes a **single chunk** added to the FAISS index alongside
the raw text chunks. It is what makes aggregation queries ("largest
purchases", "total spending", "restaurant charges") work — without it, the
model would have to read 60 raw chunks scattered across 12 files.

Because the LLM generates the summary by reading the document itself, this
works for **any document type** without code changes: credit card
statements, invoices, tax returns, bakery receipts, inventory reports.

### 3. Embed with fastembed (`inference/retriever.py`, `data_pipeline/reindex.py`)

Every chunk (raw + summary) is converted into a 384-dimensional vector
using **fastembed** running the `all-MiniLM-L6-v2` model via ONNX Runtime.

- Pure CPU, no Metal/GPU — stable on Apple Silicon and any hardware
- The same model is used at both index time and query time, so vectors
  are always comparable

### 4. Save to FAISS

All vectors are stored in a `faiss.IndexFlatIP` (inner-product / cosine
similarity index) on disk:

```
data/output/
  faiss.index     ← the vector index
  metadata.json   ← text + source info for each chunk
  centroid.npy    ← average of all chunk vectors (used by domain gate)
```

---

## Step-by-Step: Query Time

Every `POST /query` request goes through this pipeline inside
`inference/server.py`:

```
User: "What are my largest purchases?"
         │
         ▼
[1] fastembed encodes the question into a 384-dim vector
         │
         ▼
[2] FAISS searches the index for the 12 most similar chunk vectors
    (top_k: 12, relevance_threshold: 0.10 in config.yaml)
         │
         ▼
[3] Domain gate checks if the query is relevant to this deployment
    (currently disabled: similarity_threshold: -1.0)
         │
         ▼
[4] Retrieved chunks are assembled into a context string
    (budget: n_ctx - max_tokens - 350 = ~3500 tokens)
         │
         ▼
[5] Ollama receives:
      system: "You are Manoj Personal Finance's assistant. Answer only
               from the context provided. Cite source IDs."
      user:   "[context]\n\nQuestion: What are my largest purchases?"
         │
         ▼
[6] Ollama streams the answer back (~8–20 seconds on M4 CPU)
         │
         ▼
[7] Validator checks the answer cites at least one source ID
         │
         ▼
Response: { answer, sources, domain_score, latency_ms, grounded, mode }
```

---

## Key Design Decisions

### Why two embedding steps?

| Tool | Role | Why |
|------|------|-----|
| fastembed (ONNX) | Text → vector | Fast, CPU-only, no Metal crashes, same output as PyTorch sentence-transformers |
| Ollama (LLM) | Text → summary / answer | Understands meaning, handles any format, works fully offline |

They are not redundant. fastembed enables **search** (finding the right
chunks). The LLM enables **understanding** (reading chunks and generating
answers).

### Why LLM summaries at index time?

Raw transaction lines (`12/06 AMAZON MKTPL 8.81`) have no natural language.
A query like "largest purchases" will never semantically match them.

The LLM summary translates raw data into language the embedding model can
match. One summary chunk per document covers all aggregation queries
(totals, largest items, spending by category) without needing to retrieve
and process 60 raw chunks at query time.

### Why fastembed instead of sentence-transformers?

`sentence-transformers` uses PyTorch which initialises Apple's Metal GPU
backend. On M-series Macs this causes segfaults when running inside
uvicorn's async event loop. fastembed uses ONNX Runtime (CPU-only by
default) and has zero Metal dependency — completely stable.

### Why Ollama instead of llama-cpp-python?

`llama-cpp-python` with Metal enabled (`n_gpu_layers=-1`) crashes with
`Segmentation fault: 11` during inference on macOS due to Metal memory
management conflicts. Ollama wraps the same llama.cpp engine but manages
Metal memory correctly through its own daemon process — no crashes.

---

## Components Map

```
babyyoday/
├── config.yaml                  ← all tuneable settings
│
├── data_pipeline/
│   ├── chunker.py               ← PDF/text reading and splitting
│   ├── summarizer.py            ← LLM-generated summary chunks
│   ├── reindex.py               ← orchestrates full index build
│   └── watcher.py               ← hot-reloads index on new files
│
├── inference/
│   ├── server.py                ← FastAPI app, query endpoint
│   ├── retriever.py             ← fastembed + FAISS search
│   ├── domain_gate.py           ← filters off-topic queries
│   ├── context_builder.py       ← assembles chunks into prompt context
│   ├── prompt.py                ← system/user prompt templates
│   └── validator.py             ← checks answer cites sources
│
├── models/
│   └── Phi-3.1-mini-128k-instruct-Q4_K_M.gguf  ← local LLM (via Ollama)
│
├── data/
│   ├── docs/credit_cards/       ← source PDFs
│   └── output/                  ← FAISS index, metadata, centroid
│
└── Modelfile                    ← Ollama model registration
```

---

## Adding a New Domain

To point this at a completely different document set (bakery invoices, tax
returns, sales reports):

1. Drop documents into `data/docs/<your-domain>/`
2. Update `config.yaml`:
   ```yaml
   business_name: "Sweet Rise Bakery"
   business_type: "bakery"
   data:
     docs_dir: "./data/docs/bakery"
   ```
3. Reindex:
   ```bash
   python -m data_pipeline.reindex \
     --docs-dir data/docs/bakery \
     --output-dir data/output
   ```
4. Restart the server. Done.

No code changes. The LLM reads whatever documents you provide and generates
appropriate summaries automatically.

---

## Running Locally

```bash
# 1. Start Ollama (first time: register the model)
ollama create phi3-finance -f Modelfile
ollama serve > /tmp/ollama.log 2>&1 &

# 2. Activate the virtualenv
source .yoday/bin/activate

# 3. Build the index (first time or after adding documents)
python -m data_pipeline.reindex \
  --docs-dir data/docs/credit_cards \
  --output-dir data/output

# 4. Start the server
uvicorn inference.server:app --log-level info

# 5. Query
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are my largest purchases?"}' \
  | python3 -m json.tool
```
