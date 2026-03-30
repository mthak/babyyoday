from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("/app/config.yaml")
LOCAL_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

admin_app = FastAPI(title="BabyYoday Admin")


@admin_app.get("/health")
def health():
    return {"status": "ok"}

def _load_config() -> dict:
    path = CONFIG_PATH if CONFIG_PATH.exists() else LOCAL_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def _get_docs_dir(cfg: dict) -> Path:
    return Path(cfg["data"]["docs_dir"])


def _get_incoming_dir(cfg: dict) -> Path:
    return Path(cfg["data"]["watch_dir"])


def _get_log_path(cfg: dict) -> Path:
    return Path(cfg.get("logging", {}).get("query_log", "/app/data/query_log.jsonl"))


@admin_app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    cfg = _load_config()
    docs_dir = _get_docs_dir(cfg)
    docs = []
    if docs_dir.exists():
        docs = sorted(
            [
                {
                    "name": f.name,
                    "size_kb": round(f.stat().st_size / 1024, 1),
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                }
                for f in docs_dir.iterdir()
                if f.is_file()
            ],
            key=lambda d: d["name"],
        )

    log_path = _get_log_path(cfg)
    recent_queries: list[dict] = []
    if log_path.exists():
        lines = log_path.read_text().strip().split("\n")
        for line in lines[-20:]:
            try:
                recent_queries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        recent_queries.reverse()

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "business_name": cfg.get("business_name", "Unknown"),
            "docs": docs,
            "doc_count": len(docs),
            "recent_queries": recent_queries,
        },
    )


@admin_app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    cfg = _load_config()
    incoming = _get_incoming_dir(cfg)
    incoming.mkdir(parents=True, exist_ok=True)

    dest = incoming / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info("Uploaded %s → %s", file.filename, dest)
    return {"status": "ok", "filename": file.filename, "message": "File uploaded. Reindexing will start automatically."}
