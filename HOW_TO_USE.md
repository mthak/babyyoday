# How to Use BabyYoday

## How the system works — three modes

The system has three modes. You pick the one that fits your time and hardware.
All three use RAG. The LLM layer and fine-tuning are optional enhancements on top.

```
┌─────────────────────────────────────────────────────────────────────┐
│  MODE 1 — Retrieval only                                            │
│  No model file needed. Returns the raw retrieved text chunks.       │
│  Good for: testing, verifying your data is indexed correctly.       │
├─────────────────────────────────────────────────────────────────────┤
│  MODE 2 — RAG + pre-trained LLM                                     │
│  Download a .gguf model. Reads retrieved chunks, writes a natural   │
│  language answer. The model was never trained on your data.         │
│  Good for: most small businesses, quick deployment.                 │
├─────────────────────────────────────────────────────────────────────┤
│  MODE 3 — RAG + LoRA fine-tuned SLM  ← best quality                │
│  Fine-tune the LLM on your business's own Q&A pairs. The model      │
│  learns your tone, terminology, and domain vocabulary.              │
│  Good for: businesses that need the most natural, branded answers.  │
└─────────────────────────────────────────────────────────────────────┘
```

**Important**: in all three modes, the agent only ever answers from retrieved
context. LoRA fine-tuning improves *how* answers are written — not *what* the
model is allowed to know. The domain gate and RAG pipeline always enforce the
data boundary.

---

## RAG vs. fine-tuning — what each one does

| | RAG | LoRA fine-tuning |
|---|---|---|
| **Purpose** | Restrict answers to your documents | Improve tone and terminology |
| **Data boundary enforcement** | Yes — the load-bearing wall | No — model still uses pretraining memory without RAG |
| **Data update speed** | Seconds (re-embed new file) | Hours (retrain) |
| **Hardware** | Mac Mini, any VPS, CPU-only | 8–16 GB RAM minimum, MPS/GPU preferred |
| **Required?** | Always | Optional enhancement |

The `all-MiniLM-L6-v2` embedding model and the base LLM are both pre-trained
and used as-is for RAG. LoRA only adjusts a small set of adapter weights
(~20–80 MB) on top of the frozen base model.

---

## What you need

| Requirement | Mode 1 (retrieval only) | Mode 2 (RAG + LLM) | Mode 3 (RAG + fine-tuned SLM) |
|-------------|------------------------|--------------------|-----------------------------|
| Python | 3.9+ | 3.9+ | 3.9+ |
| RAM | 2 GB | 4–8 GB | 8–16 GB (during training) |
| Disk | 500 MB | 2–5 GB (model) | 10–20 GB (base model + adapter) |
| GPU / MPS | Not needed | Not needed (slower without) | Strongly recommended |
| OS | macOS, Linux | macOS, Linux | macOS (M-series), Linux + CUDA |

---

## Quick start (local, no Docker)

### 1. Clone and set up the environment

```bash
git clone <your-repo-url>
cd babyyoday

python3 -m venv .yoday
source .yoday/bin/activate          # macOS / Linux
# .yoday\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Add your business data

Put your documents in a folder. Any mix of formats works:

```
my_data/
  menu.pdf
  policies.docx
  products.csv
  faq.txt
  opening_hours.md
```

Supported formats: `.pdf` `.docx` `.txt` `.md` `.csv`

> **CSV note**: Rows are converted to readable sentences.
> `name: Chocolate Cake | price: $32 | category: vegan` — each row becomes searchable.

### 3. Index your data

```bash
python setup_local.py
```

Or point it at your own data directory:

```bash
python -m data_pipeline.reindex \
  --docs-dir ./my_data/ \
  --output-dir ./data/
```

This creates three files in `./data/`:
- `faiss.index` — the vector index (your data, searchable)
- `metadata.json` — chunk text + source info
- `centroid.npy` — the domain gate centroid (what "in domain" looks like)

### 4. Set your business name

Edit `config.yaml`:

```yaml
business_name: "My Business Name"
business_type: "bakery"   # or: law, saas, gym, retail — any label
```

### 5. Start the agent

```bash
# Terminal 1 — the agent API
uvicorn inference.server:app --port 8000 --reload

# Terminal 2 — the admin panel (optional)
uvicorn admin.app:admin_app --port 8001 --reload
```

### 6. Ask it a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Do you have vegan options?"}'
```

Response:

```json
{
  "answer": "[Retrieval-only mode] Found 4 chunks. Sources: DOC-6AE12C-1, ...",
  "sources": [
    { "id": "DOC-6AE12C-1", "name": "allergens.txt", "score": 0.509 }
  ],
  "domain_score": 0.36,
  "latency_ms": 252,
  "grounded": true
}
```

> **Retrieval-only mode**: Without a `.gguf` model file, the system returns the
> retrieved chunks directly. Add a model (step 7) to get full natural-language answers.

### 7. Add a language model (Mode 2 — RAG + pre-trained LLM)

Download a quantized GGUF model and place it at `./models/model.gguf`.
The server auto-detects the model family from the filename and applies
the correct chat template (Phi-3, Mistral, or Llama-3 format).

```bash
mkdir -p models

# Option A — Phi-3-mini (3.8B, ~2.5 GB, runs on 4 GB RAM)
# Download from: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf

# Option B — Mistral-7B (7B, ~4.5 GB, runs on 8 GB RAM)
# Download from: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

# Option C — Llama-3-8B (8B, ~5 GB, best quality)
# Download from: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

mv ~/Downloads/your-model.gguf ./models/model.gguf
```

Restart the server — it detects and loads the model automatically.

The response now includes a `"mode"` field:
- `"mode": "llm"` — full natural-language answer from the LLM
- `"mode": "retrieval-only"` — no model loaded, returns raw chunks

**Model selection guide:**

| Model | Size | RAM needed | Hardware |
|-------|------|-----------|----------|
| Phi-3-mini | 3.8B | ~3 GB | Mac Mini 8 GB, $5 Lightsail |
| Mistral-7B | 7B | ~5 GB | Mac Mini 16 GB, $10 Lightsail |
| Llama-3-8B | 8B | ~6 GB | Mac Mini 16 GB+, $20 Lightsail |

---

## Running with Docker (for deployment)

### Build a customer's container

```bash
python builder/build_customer.py \
  --business-name "Sweet Rise Bakery" \
  --business-type bakery \
  --data ./my_data/ \
  --model-path ./models/model.gguf \
  --tag sweetrise-agent:latest
```

This single command:
1. Chunks and embeds all documents in `./my_data/`
2. Builds the FAISS index
3. Computes the domain gate centroid
4. Copies the model
5. Builds a self-contained Docker image

### Run the container

```bash
docker run -p 8000:8000 -p 8001:8001 sweetrise-agent:latest
```

The container is now live on:
- `http://localhost:8000` — agent API
- `http://localhost:8001` — admin panel

### Deploy to a Mac Mini or AWS Lightsail

```bash
# Copy the image to the customer's machine
docker save sweetrise-agent:latest | gzip > sweetrise-agent.tar.gz
scp sweetrise-agent.tar.gz user@customer-machine:~

# On the customer's machine
docker load < sweetrise-agent.tar.gz
docker run -d --restart=always -p 8000:8000 -p 8001:8001 sweetrise-agent:latest
```

---

## LoRA fine-tuning (Mode 3 — RAG + fine-tuned SLM)

Fine-tuning teaches the model your business's language, tone, and terminology.
It does **not** replace RAG — both run together. The fine-tuned model still
only answers from retrieved context, but its answers feel more natural and
on-brand than a generic pre-trained model.

### How the full pipeline works

```
Your documents
      │
      ▼
[generate_training_data.py]   Auto-creates Q&A pairs from every chunk
      │                        + uses your eval_dataset.yaml as extra examples
      │  data/training_data.jsonl
      ▼
[train_lora.py]               QLoRA fine-tuning on Phi-3 / Mistral / Llama-3
      │                        Runs on Mac M-series (30 min) or CPU (2 h)
      │  models/lora_adapter/  (~20–80 MB)
      ▼
[merge_adapter.py]            Merges adapter into base model
      │                        Exports to GGUF via llama.cpp
      │  models/model.gguf
      ▼
[build_customer.py]           Bundles everything into the Docker container
```

### Step 1 — Install training dependencies

```bash
source .yoday/bin/activate
pip install transformers peft accelerate bitsandbytes datasets trl
```

### Step 2 — Generate training data from your documents

```bash
python model_training/generate_training_data.py \
  --docs-dir ./data/docs/ \
  --output ./data/training_data.jsonl \
  --business-name "Sweet Rise Bakery" \
  --manual-qa evaluator/eval_dataset.yaml
```

This reads every document chunk and auto-generates question/answer training
pairs using template patterns. It also pulls in your manually curated Q&A
pairs from `eval_dataset.yaml` for higher quality examples.

Output is a `.jsonl` file — one training example per line:

```json
{"system": "You are Sweet Rise Bakery's assistant...",
 "user": "Context: [chunk text]\n\nQuestion: Do you have vegan cakes?",
 "assistant": "Yes, we have the Chocolate Avocado Cake and Coconut Berry Tart..."}
```

### Step 3 — Fine-tune with LoRA

Choose a config that matches your base model and hardware:

| Config | Base model | Hardware | Train time |
|--------|-----------|---------|-----------|
| `phi3_lora.yaml` | Phi-3-mini-4k-instruct | 8 GB RAM, M2 Mac | ~30 min |
| `mistral_lora.yaml` | Mistral-7B-Instruct-v0.2 | 16 GB RAM, M2 Mac | ~1 h |
| `llama3_lora.yaml` | Meta-Llama-3-8B-Instruct | 16 GB RAM + GPU | ~1 h |

```bash
python model_training/train_lora.py \
  --config model_training/configs/phi3_lora.yaml
```

Or override specific values without editing the config:

```bash
python model_training/train_lora.py \
  --config model_training/configs/phi3_lora.yaml \
  --epochs 5 \
  --batch-size 4
```

The adapter is saved to `./models/phi3_lora_adapter/` (~20–80 MB).
The base model weights are never modified — only the adapter changes.

### Step 4 — Merge adapter + export to GGUF

```bash
python model_training/merge_adapter.py \
  --adapter-path ./models/phi3_lora_adapter/ \
  --output-dir   ./models/merged/ \
  --gguf-output  ./models/model.gguf \
  --quantization q4_k_m
```

Quantization options (affects final model size and quality):

| Option | Size | Quality | Use when |
|--------|------|---------|---------|
| `q4_k_m` | ~2.5 GB | Good | Default — Mac Mini, small VPS |
| `q5_k_m` | ~3.5 GB | Better | More RAM available |
| `q8_0`   | ~6 GB  | Best   | Maximum quality, large instance |

> **Requires llama.cpp**: The GGUF export step needs llama.cpp installed.
> If it's not found, the script prints exact manual conversion instructions.
> Install it with: `git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make`

### Step 5 — Start the server with your fine-tuned model

```bash
uvicorn inference.server:app --port 8000 --reload
```

The server loads `./models/model.gguf` automatically. Check the health endpoint:

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "business": "Sweet Rise Bakery",
  "model_loaded": true,
  "model_path": "./models/model.gguf"
}
```

### One-command build with fine-tuning (Docker)

To run the entire pipeline (data → training → merge → Docker image) in one command:

```bash
python builder/build_customer.py \
  --business-name "Sweet Rise Bakery" \
  --business-type bakery \
  --data ./sample_data/ \
  --lora-config model_training/configs/phi3_lora.yaml \
  --tag sweetrise-agent:latest
```

This runs all four steps automatically and produces a container with the
fine-tuned SLM baked in. Compare with the RAG-only build:

```bash
# RAG + pre-trained LLM (fast, no training)
python builder/build_customer.py \
  --business-name "Sweet Rise Bakery" \
  --data ./sample_data/ \
  --model-path ./models/model.gguf \
  --tag sweetrise-agent:latest

# RAG + fine-tuned SLM (best quality, takes 30 min–2 h)
python builder/build_customer.py \
  --business-name "Sweet Rise Bakery" \
  --data ./sample_data/ \
  --lora-config model_training/configs/phi3_lora.yaml \
  --tag sweetrise-agent:latest
```

### When to re-run fine-tuning

You don't need to retrain every time data changes. Retraining makes sense when:

- The business has accumulated 50+ real customer Q&A interactions (use as training data)
- Answers feel robotic or don't match the business's tone
- New product lines or services have been added with different terminology

For day-to-day data updates (new prices, updated hours, new menu items), just
re-embed the documents — no retraining needed.

### Troubleshooting fine-tuning

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Reduce `batch_size` to 1, increase `gradient_accumulation_steps` to 8 |
| `bitsandbytes` not working on Mac | Set `load_in_4bit: false` in the config — trains in fp16 instead |
| Training loss not decreasing | Increase `epochs` to 5, or lower `learning_rate` to `1e-4` |
| GGUF conversion fails | Install llama.cpp (see instructions printed by merge_adapter.py) |
| Model answers feel generic | Add more hand-written Q&A pairs to `eval_dataset.yaml` and regenerate |

---

## Keeping data up to date

### Option 1 — File watcher (automatic)

While the container is running, drop any new file into `/app/data/incoming/`.
The watcher detects it, embeds it, and updates the index automatically.

```bash
# Local dev
cp new_menu.pdf ./data/incoming/
# The watcher picks it up within seconds
```

### Option 2 — Admin panel upload

Open `http://localhost:8001` in a browser.
Use the "Upload New Data" section to upload a file.
The index updates automatically after upload.

### Option 3 — Manual reindex

```bash
source .yoday/bin/activate
python -m data_pipeline.reindex \
  --docs-dir ./data/docs/ \
  --output-dir ./data/
```

Then restart the server, or call the `/reload` endpoint (if running hot-reload mode).

---

## Evaluate the agent

Run the built-in evaluator to see how well the agent is performing:

```bash
python evaluator/runner.py
```

Output:

```
============================================================
  AGENT EVALUATION REPORT
============================================================
  Total test cases       : 17
  Domain gate accuracy   : 88.2%
  Retrieval hit rate     : 83.3%
  Avg source recall      : 100.0%
  Avg top chunk score    : 0.517
  Avg answer groundedness: 91.9%
  Avg citation coverage  : 20.9%
  Avg semantic similarity: 0.529
  Avg latency            : 328 ms
============================================================
```

### What each metric means

| Metric | What it measures | Good value |
|--------|-----------------|-----------|
| **Domain gate accuracy** | % of queries correctly allowed or rejected | > 90% |
| **Retrieval hit rate** | % of valid queries that found relevant chunks | > 85% |
| **Avg source recall** | Did retrieval find the right documents? | > 80% |
| **Avg top chunk score** | How similar is the best chunk to the query? | > 0.45 |
| **Answer groundedness** | % of answer words found in the retrieved context | > 85% |
| **Citation coverage** | % of answer sentences that cite a source | > 50% (with LLM) |
| **Semantic similarity** | How similar is the answer to the expected answer? | > 0.6 |

### Add your own test cases

Edit `evaluator/eval_dataset.yaml`:

```yaml
cases:
  - id: "MY-001"
    category: "menu"
    query: "Do you have sourdough bread?"
    expected_pass: true                  # should the agent answer this?
    expected_sources: ["menu.txt"]       # which docs should be retrieved?
    expected_answer: "Yes, we sell sourdough loaf for $7."

  - id: "MY-OUT-001"
    category: "off_topic"
    query: "What is the weather today?"
    expected_pass: false                 # should be rejected
    expected_sources: []
    expected_answer: null
```

Run and save a JSON report:

```bash
python evaluator/runner.py --output eval_report.json
```

### Fixing poor eval scores

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| Gate rejecting valid queries | Threshold too high | Lower `domain_gate.similarity_threshold` in `config.yaml` (try 0.15) |
| Gate passing off-topic queries | Threshold too low | Raise `domain_gate.similarity_threshold` (try 0.30) |
| Low retrieval hit rate | Not enough documents | Add more docs covering that topic |
| Low groundedness | LLM generating from memory | Tighten system prompt in `inference/prompt.py` |
| Low source recall | Relevant content split across chunks | Reduce `chunking.chunk_size` in `config.yaml` |

---

## API reference

### `POST /query`

Ask the agent a question.

```json
{ "query": "Do you have gluten-free options?" }
```

Response (success):

```json
{
  "answer": "Yes, the Gluten-Free Chocolate Cupcake and Coconut Berry Tart are gluten-free. [DOC-6AE12C-1]",
  "sources": [
    { "id": "DOC-6AE12C-1", "name": "allergens.txt", "score": 0.62 },
    { "id": "DOC-3F0716-0", "name": "menu.txt",      "score": 0.55 }
  ],
  "domain_score": 0.61,
  "latency_ms": 340,
  "grounded": true
}
```

Response (rejected — off-topic):

```json
{
  "error": "I can only help with questions about Sweet Rise Bakery. How can I help with that?",
  "domain_score": 0.08
}
```

### `GET /health`

```json
{
  "status": "ok",
  "business": "Sweet Rise Bakery",
  "model_loaded": true,
  "index_size": 10
}
```

---

## Project structure

```
babyyoday/
│
├── inference/          Core agent pipeline
│   ├── server.py         FastAPI server — main entry point
│   ├── retriever.py      FAISS search — finds relevant chunks
│   ├── domain_gate.py    Rejects off-topic queries before they reach the LLM
│   ├── context_builder.py  Assembles retrieved chunks into a prompt context
│   ├── prompt.py         Builds the system + user prompt
│   └── validator.py      Checks citations in the answer
│
├── agent/              Multi-step reasoning
│   ├── planner.py        Splits compound questions into sub-tasks
│   ├── executor.py       Runs each sub-task against retrieval + LLM
│   └── router.py         Merges sub-task answers into one response
│
├── data_pipeline/      Data ingestion
│   ├── chunker.py        Reads PDF/DOCX/CSV/TXT → text chunks
│   ├── reindex.py        Embeds chunks → builds FAISS index
│   └── watcher.py        Watches a folder for new files → auto-reindex
│
├── builder/            Customer container factory
│   ├── build_customer.py  End-to-end: data → Docker image
│   ├── embed_data.py      Standalone embedding script
│   ├── build_gate.py      Compute domain centroid
│   ├── Dockerfile         Container definition
│   └── config_template.yaml
│
├── admin/              Simple web UI
│   ├── app.py            FastAPI admin panel
│   └── templates/        HTML dashboard
│
├── evaluator/          Performance measurement
│   ├── metrics.py        Groundedness, recall, gate accuracy, semantic sim
│   ├── runner.py         Runs all test cases, prints report
│   └── eval_dataset.yaml  Ground truth Q&A pairs
│
├── model_training/     Optional LoRA fine-tuning pipeline
│   ├── generate_training_data.py  Auto-generates Q&A pairs from your documents
│   ├── train_lora.py              QLoRA fine-tuning (PEFT + TRL)
│   ├── merge_adapter.py           Merges adapter → GGUF for llama.cpp
│   └── configs/
│       ├── phi3_lora.yaml         Config for Phi-3-mini (Mac-friendly)
│       ├── mistral_lora.yaml      Config for Mistral-7B
│       └── llama3_lora.yaml       Config for Llama-3-8B
│
├── tests/              Unit tests (25 tests, all passing)
├── sample_data/        Example bakery documents for testing
├── config.yaml         Business configuration
├── setup_local.py      One-command local setup
└── docker-compose.yml  Local Docker dev environment
```

---

## Troubleshooting

**"Model not found — running in retrieval-only mode"**
→ No `.gguf` file at `./models/model.gguf`. Download a model (step 7), run fine-tuning
  (Mode 3), or ignore this — retrieval-only mode still finds the right documents.

**Valid queries being rejected by the domain gate**
→ Lower `domain_gate.similarity_threshold` in `config.yaml`. Run
  `python evaluator/runner.py` to see which queries are failing and their scores.

**Low retrieval scores (top score < 0.3)**
→ The query wording is too different from the document wording. Try rephrasing the
  question, or add a FAQ document that uses customer-style language.

**Admin panel shows no documents**
→ Check that `data.docs_dir` in `config.yaml` points to the right folder.

**Container runs out of RAM**
→ Switch to a smaller model (Phi-3-mini instead of Mistral-7B), or increase
  the instance size. See the hardware table in the deployment section.
