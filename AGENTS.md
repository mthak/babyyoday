# AGENTS.md

## Cursor Cloud specific instructions

BabyYoday is a single Python project (FastAPI + FAISS RAG). There is no npm/frontend app. See `HOW_TO_USE.md` and `README.md` for full architecture.

### One-time VM prerequisites

Ubuntu images may lack `python3-venv`. Install before creating the venv:

```bash
sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3.12-venv build-essential
```

### Python environment

Use the project venv at `.yoday` (not committed):

```bash
python3 -m venv .yoday
source .yoday/bin/activate
pip install -r requirements.txt fastembed
```

`fastembed` is required at runtime (`inference/retriever.py`, `data_pipeline/reindex.py`) but is not listed in `requirements.txt` yet.

### Index data before starting the API

`config.yaml` expects the FAISS index under `./data/output/`, while `setup_local.py` writes to `./data/`:

```bash
python setup_local.py
mkdir -p data/output
cp data/faiss.index data/metadata.json data/centroid.npy data/output/
```

For your own documents, use `python -m data_pipeline.reindex --docs-dir ./my_data/ --output-dir ./data/output`.

### Running services

| Service | Command | URL |
|---------|---------|-----|
| Inference API (required) | `uvicorn inference.server:app --host 0.0.0.0 --port 8000` | `http://localhost:8000` |
| Admin panel (optional) | `uvicorn admin.app:admin_app --host 0.0.0.0 --port 8001` | `http://localhost:8001` |
| Ollama LLM (optional) | `ollama serve` + `ollama create phi3-finance -f Modelfile` | `http://localhost:11434` |

Without Ollama, the API runs in **retrieval-only mode** (`mode: "retrieval-only"` in `/query` responses). This is expected for local dev.

### Verify the stack

```bash
pytest tests/ -v
curl http://localhost:8000/health
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "Do you have vegan options?"}'
```

There is no configured linter (ruff/flake8/mypy). CI runs pytest only (`.github/workflows/build.yml`).

### Gotchas

- `config.yaml` is set for "Manoj Personal Finance" (`data/docs/credit_cards`); sample bakery docs from `setup_local.py` land in `data/docs/`. Queries still work against the indexed sample data; the file watcher path may not match until you align paths or reindex.
- `HF_HUB_OFFLINE=1` is set in `inference/server.py`; first run still downloads the fastembed ONNX model from Hugging Face during indexing.
- Admin dashboard (`:8001`) may error with recent Starlette/Jinja2 (`TypeError: unhashable type: 'dict'` on `/`); the inference API on `:8000` is the primary dev surface.
- Docker alternative: `docker compose up --build` (does not include Ollama).
