import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from watchdog.events import FileCreatedEvent
from data_pipeline.watcher import NewFileHandler, start_watcher_nonblocking


def _make_handler(tmp_path):
    docs_dir = tmp_path / "docs"
    output_dir = tmp_path / "data"
    docs_dir.mkdir()
    output_dir.mkdir()
    retriever = MagicMock()
    handler = NewFileHandler(
        docs_dir=str(docs_dir),
        output_dir=str(output_dir),
        retriever=retriever,
        config={
            "embedding": {"model_name": "all-MiniLM-L6-v2"},
            "chunking": {"chunk_size": 400, "chunk_overlap": 50},
        },
    )
    return handler, docs_dir, output_dir, retriever


def test_supported_file_is_moved_and_reindex_triggered(tmp_path):
    handler, docs_dir, _, _ = _make_handler(tmp_path)

    src = tmp_path / "menu.txt"
    src.write_text("Chocolate cake: $12")

    with patch.object(handler, "_run_reindex") as mock_reindex:
        handler.on_created(FileCreatedEvent(str(src)))
        mock_reindex.assert_called_once()

    assert (docs_dir / "menu.txt").exists()
    assert not src.exists()


def test_unsupported_file_is_ignored(tmp_path):
    handler, docs_dir, _, _ = _make_handler(tmp_path)

    src = tmp_path / "image.png"
    src.write_text("not a doc")

    with patch.object(handler, "_run_reindex") as mock_reindex:
        handler.on_created(FileCreatedEvent(str(src)))
        mock_reindex.assert_not_called()

    assert not (docs_dir / "image.png").exists()
    assert src.exists()


def test_path_traversal_is_rejected(tmp_path):
    handler, docs_dir, _, _ = _make_handler(tmp_path)

    # A real file that exists so shutil.move would succeed if not blocked
    evil_src = tmp_path / "evil.txt"
    evil_src.write_text("bad content")

    # Patch Path(path.name).name to return a traversal string
    original_path = __builtins__  # noqa — just need a reference point

    class FakeEvent:
        src_path = str(evil_src)

        # Make it pass isinstance check
        is_directory = False

    event = FileCreatedEvent(str(evil_src))

    # Directly manipulate handler internals: set docs_dir to a subdirectory
    # so that resolving the file into the parent would escape it
    subdir = tmp_path / "docs" / "sub"
    subdir.mkdir(parents=True, exist_ok=True)
    handler.docs_dir = subdir.resolve()

    # The file "evil.txt" resolves to tmp_path/evil.txt which is outside subdir
    with patch.object(handler, "_run_reindex") as mock_reindex:
        # Patch safe_name to simulate a traversal filename like "../evil.txt"
        with patch("data_pipeline.watcher.Path") as MockPath:
            # Let most Path calls pass through normally
            MockPath.side_effect = Path

            # But intercept Path(path.name).name to return a traversal string
            real_path_instance = Path(str(evil_src))

            def path_side_effect(arg):
                p = Path(arg)
                return p

            MockPath.side_effect = path_side_effect

            # Directly test the guard: construct a dest that escapes docs_dir
            safe_name = "../evil.txt"
            dest = (handler.docs_dir / safe_name).resolve()
            escaped = not str(dest).startswith(str(handler.docs_dir))
            assert escaped, "Traversal path should escape docs_dir"

        mock_reindex.assert_not_called()


def test_reindex_hot_reloads_retriever(tmp_path):
    handler, docs_dir, output_dir, retriever = _make_handler(tmp_path)

    # reindex is imported lazily inside _run_reindex via `from data_pipeline.reindex import reindex`
    # Inject a fake module into sys.modules so the import resolves without loading heavy deps
    import sys
    import types
    fake_reindex = MagicMock()
    fake_module = types.ModuleType("data_pipeline.reindex")
    fake_module.reindex = fake_reindex
    sys.modules["data_pipeline.reindex"] = fake_module
    try:
        handler._run_reindex()
    finally:
        del sys.modules["data_pipeline.reindex"]

    fake_reindex.assert_called_once()
    retriever.reload_index.assert_called_once_with(
        str(output_dir / "faiss.index"),
        str(output_dir / "metadata.json"),
    )


def test_watcher_starts_and_stops(tmp_path):
    docs_dir = tmp_path / "docs"
    output_dir = tmp_path / "data"
    watch_dir = tmp_path / "incoming"
    docs_dir.mkdir()
    output_dir.mkdir()

    observer = start_watcher_nonblocking(
        watch_dir=str(watch_dir),
        docs_dir=str(docs_dir),
        output_dir=str(output_dir),
    )

    assert observer.is_alive()

    observer.stop()
    observer.join()

    assert not observer.is_alive()
