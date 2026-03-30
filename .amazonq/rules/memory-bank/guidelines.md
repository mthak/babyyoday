# BabyYoday — Development Guidelines

## Code Quality Standards

### Universal File Header
Every Python module starts with `from __future__ import annotations` — used in all 5 analyzed files without exception.

### Module-Level Logger
Every module declares a logger immediately after imports:
```python
logger = logging.getLogger(__name__)
```
Never use `print()` for runtime output. All status, warnings, and errors go through the logger.

### Logging Format
- Info: progress milestones and counts — `logger.info("Chunked %s → %d chunks", path.name, len(result))`
- Warning: degraded-mode situations that don't crash — `logger.warning("Model not found at %s — running in retrieval-only mode", model_path)`
- Error: fatal failures before `sys.exit(1)` — `logger.error("Docker build failed:\n%s", result.stderr)`
- Always use `%s` / `%d` lazy formatting, never f-strings in logger calls

### Type Annotations
- All function signatures are fully annotated (parameters + return types)
- Use `str | Path` union types for filesystem arguments (accept both, convert internally with `Path()`)
- Use `list[Type]` lowercase generics (Python 3.10+ style), not `List[Type]` from `typing`
- `Optional[X]` from `typing` is used for nullable return types in public APIs; `X | None` is acceptable too
- Dataclasses use field-level annotations without defaults unless needed

### Dataclasses for Data Transfer
Structured data is always a `@dataclass`, never a plain dict or tuple:
```python
@dataclass
class Chunk:
    text: str
    source_id: str
    source_name: str
    chunk_index: int
```
Used for: `Chunk`, `RetrievedChunk`, `ValidationResult`. Pydantic `BaseModel` is used only for FastAPI request/response schemas.

---

## Structural Conventions

### Config Loading Pattern
Both `server.py` and `admin/app.py` use the same dual-path config pattern — container path first, local dev fallback second:
```python
CONFIG_PATH = Path("/app/config.yaml")
LOCAL_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

def load_config() -> dict:
    path = CONFIG_PATH if CONFIG_PATH.exists() else LOCAL_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)
```
Never hardcode config values inline. Always read from `config.yaml`.

### Path Handling
Always use `pathlib.Path`, never `os.path`:
```python
from pathlib import Path
BUILDER_DIR = Path(__file__).parent
PROJECT_ROOT = BUILDER_DIR.parent
```
Convert string arguments to `Path` at the function boundary, not at call sites.

### Graceful Degradation
Components that may be unavailable (model, FAISS index, domain gate) are loaded with try/except in the lifespan handler and stored as `None`. Callers check for `None` before use:
```python
if retriever is None:
    return ErrorResponse(error="Index not ready yet.")
```
The system runs in "retrieval-only mode" without a model, and "index-pending mode" without a FAISS index. Never crash on missing optional components.

### FastAPI Application Structure
- Use `lifespan` context manager (not deprecated `on_event`) for startup/shutdown
- Store all shared state in a module-level `_state: dict = {}` dict
- Pydantic models for all request/response schemas — never raw dicts in endpoint signatures
- Health endpoint always returns `{"status": "ok"}` or a richer dict with component readiness

```python
_state: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # load components into _state
    yield
    # cleanup

app = FastAPI(title="BabyYoday Agent", lifespan=lifespan)
```

### Class Design
Classes are used for stateful components that hold loaded resources (Retriever, DomainGate). Pure transformation logic uses module-level functions, not classes.

---

## Naming Conventions

| Pattern | Convention | Example |
|---------|-----------|---------|
| Module-level constants | `UPPER_SNAKE_CASE` | `CONFIG_PATH`, `SUPPORTED_EXTENSIONS`, `BUILDER_DIR` |
| Private helpers | `_snake_case` prefix | `_load_config()`, `_read_pdf()`, `_make_source_id()` |
| Public functions | `snake_case` | `process_file()`, `chunk_text()`, `validate_response()` |
| Classes | `PascalCase` | `Retriever`, `DomainGate`, `QueryRequest` |
| Dataclass fields | `snake_case` | `source_id`, `chunk_index`, `is_valid` |
| CLI arguments | `--kebab-case` | `--business-name`, `--model-path`, `--lora-config` |

Private helpers that are only used within a module are prefixed with `_`. Functions intended for import by other modules have no prefix.

---

## Semantic Patterns

### Dispatch by File Extension
Use a dict or if/elif chain to dispatch to format-specific readers, with a fallback warning:
```python
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".csv"}

def read_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return _read_text_file(path)
    elif ext == ".csv":
        return _read_csv(path)
    elif ext == ".pdf":
        return _read_pdf(path)
    elif ext == ".docx":
        return _read_docx(path)
    else:
        logger.warning("Unsupported file type: %s", ext)
        return ""
```

### Optional Import Pattern
Heavy dependencies (pypdf, python-docx, llama_cpp) are imported inside functions with `try/except ImportError`, returning empty string or `None` and logging a warning. This keeps the module importable even if the dependency is missing:
```python
def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
        ...
    except ImportError:
        logger.warning("pypdf not installed — skipping %s", path.name)
        return ""
```

### Subprocess Orchestration
Multi-step build pipelines use `subprocess.run()` with `capture_output=False` (so output streams to terminal) and check `returncode != 0` to fail fast:
```python
result = subprocess.run([sys.executable, script, "--arg", value], capture_output=False)
if result.returncode != 0:
    logger.error("Step failed")
    sys.exit(1)
```
Use `capture_output=True, text=True` only when you need to inspect stderr (e.g., Docker build errors).

### Section Comments in Long Functions
Long functions use `# ── Description ──────` separator comments to visually divide logical phases:
```python
# ── 1. Copy customer documents ────────────────────────────────────────────
# ── 2. Build FAISS index ──────────────────────────────────────────────────
# ── 3. Resolve model ─────────────────────────────────────────────────────
```
This pattern appears in both `build_customer.py` and CDK stacks.

### Source ID Generation
Chunk source IDs are deterministic, human-readable, and collision-resistant:
```python
def _make_source_id(path: Path, chunk_idx: int) -> str:
    name_hash = hashlib.md5(path.name.encode()).hexdigest()[:6].upper()
    return f"DOC-{name_hash}-{chunk_idx}"
```
Format: `DOC-<6-char-hash>-<index>` (e.g., `DOC-A3F2C1-0`).

### Validation Result Pattern
Validators return a dataclass with both a boolean flag and the (possibly modified) data, never raising exceptions for validation failures:
```python
@dataclass
class ValidationResult:
    is_valid: bool
    cited_sources: list[str]
    unknown_sources: list[str]
    answer: str  # original answer, unmodified
```

### Config-Driven Thresholds
All numeric thresholds (similarity, relevance, chunk size, top_k) come from `config.yaml`. Never hardcode them. Pass them as constructor arguments to classes:
```python
Retriever(
    index_path=cfg["faiss"]["index_path"],
    top_k=cfg["retrieval"]["top_k"],
    relevance_threshold=cfg["retrieval"]["relevance_threshold"],
)
```

---

## CDK Infrastructure Patterns (TypeScript)

### Stack Interface Props
Every CDK stack defines a typed `interface XxxStackProps extends cdk.StackProps` for cross-stack dependencies. Resources are passed as typed props, never looked up by name at deploy time.

### Section Comments
CDK stacks use `// ── Section name ──────────────────────────────────────────────────────────` to separate logical resource groups (security groups, task definitions, ALB, etc.).

### Removal Policy
Log groups use `RemovalPolicy.DESTROY`. S3 and EFS use `RemovalPolicy.RETAIN` to prevent data loss on `cdk destroy`.

### CfnOutput for Discoverability
Every stack exports its key resource identifiers via `new cdk.CfnOutput(...)` so they appear in the CloudFormation console after deploy.

---

## Testing Conventions

Test files mirror the module they test: `tests/test_chunker.py` tests `data_pipeline/chunker.py`. Use `pytest` with no special framework beyond standard fixtures. Test files are in `tests/` at the project root.
