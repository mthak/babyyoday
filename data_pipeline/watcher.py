from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler, FileCreatedEvent
from watchdog.observers import Observer

from data_pipeline.chunker import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class NewFileHandler(FileSystemEventHandler):
    """When a new file lands in the watch dir, move it to docs and trigger reindex."""

    def __init__(self, docs_dir: str, output_dir: str, retriever=None, config: dict | None = None):
        self.docs_dir = Path(docs_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.retriever = retriever
        self.config = config or {}
        self._reindex_pending = False

    def on_created(self, event):
        if not isinstance(event, FileCreatedEvent):
            return

        path = Path(event.src_path)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.info("Ignoring unsupported file: %s", path.name)
            return

        # Strip any directory components from the filename to prevent path traversal
        safe_name = Path(path.name).name
        dest = (self.docs_dir / safe_name).resolve()
        if not str(dest).startswith(str(self.docs_dir)):
            logger.warning("Rejected path traversal attempt: %s", path.name)
            return

        logger.info("New file detected: %s → moving to %s", safe_name, dest)
        shutil.move(str(path), str(dest))
        self._run_reindex()

    def _run_reindex(self):
        from data_pipeline.reindex import reindex

        logger.info("Triggering reindex ...")
        reindex(
            docs_dir=str(self.docs_dir),
            output_dir=str(self.output_dir),
            embedding_model_name=self.config.get("embedding", {}).get(
                "model_name", "all-MiniLM-L6-v2"
            ),
            chunk_size=self.config.get("chunking", {}).get("chunk_size", 400),
            chunk_overlap=self.config.get("chunking", {}).get("chunk_overlap", 50),
        )

        if self.retriever is not None:
            index_path = str(self.output_dir / "faiss.index")
            metadata_path = str(self.output_dir / "metadata.json")
            self.retriever.reload_index(index_path, metadata_path)
            logger.info("Retriever hot-reloaded with new index")


def start_watcher_nonblocking(
    watch_dir: str,
    docs_dir: str,
    output_dir: str,
    retriever=None,
    config: dict | None = None,
) -> Observer:
    """Start the file watcher and return the observer (non-blocking)."""
    watch_path = Path(watch_dir)
    watch_path.mkdir(parents=True, exist_ok=True)

    handler = NewFileHandler(docs_dir, output_dir, retriever, config)
    observer = Observer()
    observer.schedule(handler, str(watch_path), recursive=False)
    observer.start()
    logger.info("Watching %s for new files ...", watch_path)
    return observer


def start_watcher(
    watch_dir: str,
    docs_dir: str,
    output_dir: str,
    retriever=None,
    config: dict | None = None,
):
    """Start watching a directory for new files. Blocking call."""
    observer = start_watcher_nonblocking(watch_dir, docs_dir, output_dir, retriever, config)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Watch for new documents")
    parser.add_argument("--watch-dir", required=True)
    parser.add_argument("--docs-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    start_watcher(args.watch_dir, args.docs_dir, args.output_dir, config=cfg)
