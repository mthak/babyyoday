#!/usr/bin/env python3
"""
Build a customer's Docker image — with optional LoRA fine-tuning.

Two modes:

  RAG-only (fast, no training):
    python builder/build_customer.py \
        --business-name "Sweet Rise Bakery" \
        --data ./sample_data/ \
        --model-path ./models/phi-3-mini.gguf \
        --tag sweetrise-agent:latest

  RAG + fine-tuned SLM (best quality, takes 30min-2h):
    python builder/build_customer.py \
        --business-name "Sweet Rise Bakery" \
        --data ./sample_data/ \
        --base-model microsoft/Phi-3-mini-4k-instruct \
        --lora-config model_training/configs/phi3_lora.yaml \
        --tag sweetrise-agent:latest
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.reindex import reindex

logger = logging.getLogger(__name__)

BUILDER_DIR = Path(__file__).parent
PROJECT_ROOT = BUILDER_DIR.parent


def build_config(business_name: str, business_type: str) -> dict:
    with open(BUILDER_DIR / "config_template.yaml") as f:
        template = f.read()
    rendered = template.replace("{{ business_name }}", business_name)
    rendered = rendered.replace("{{ business_type }}", business_type)
    return yaml.safe_load(rendered)


def run_fine_tuning(
    business_name: str,
    docs_dir: Path,
    base_model: str,
    lora_config_path: str,
    staging: Path,
) -> Path:
    """Run the full fine-tuning pipeline and return path to the final GGUF."""

    training_data_path = staging / "data" / "training_data.jsonl"
    adapter_path       = staging / "models" / "lora_adapter"
    merged_path        = staging / "models" / "merged"
    gguf_path          = staging / "models" / "model.gguf"

    # Step 1: Generate training data from the business's documents
    logger.info("── Step 1/3: Generating training data ──")
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "model_training" / "generate_training_data.py"),
            "--docs-dir",      str(docs_dir),
            "--output",        str(training_data_path),
            "--business-name", business_name,
            "--manual-qa",     str(PROJECT_ROOT / "evaluator" / "eval_dataset.yaml"),
        ],
        capture_output=False,
    )
    if result.returncode != 0:
        logger.error("Training data generation failed")
        sys.exit(1)

    # Step 2: Fine-tune with LoRA
    logger.info("── Step 2/3: LoRA fine-tuning ──")
    lora_cfg_overrides = {
        "training_data": str(training_data_path),
        "output_dir":    str(adapter_path),
    }
    # Write a modified config with the staging paths
    lora_cfg_staging = staging / "lora_config.yaml"
    with open(lora_config_path) as f:
        lora_cfg = yaml.safe_load(f)
    lora_cfg.update(lora_cfg_overrides)
    if base_model:
        lora_cfg["base_model"] = base_model
    with open(lora_cfg_staging, "w") as f:
        yaml.dump(lora_cfg, f)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "model_training" / "train_lora.py"),
            "--config", str(lora_cfg_staging),
        ],
        capture_output=False,
    )
    if result.returncode != 0:
        logger.error("LoRA training failed")
        sys.exit(1)

    # Step 3: Merge adapter + export to GGUF
    logger.info("── Step 3/3: Merging adapter + exporting GGUF ──")
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "model_training" / "merge_adapter.py"),
            "--adapter-path", str(adapter_path),
            "--output-dir",   str(merged_path),
            "--gguf-output",  str(gguf_path),
        ],
        capture_output=False,
    )
    if result.returncode != 0:
        logger.warning("GGUF export failed — check if llama.cpp is installed")
        logger.warning("You can convert manually and re-run build without --lora-config")

    if gguf_path.exists():
        logger.info("Fine-tuned model ready: %s", gguf_path)
        return gguf_path
    else:
        logger.warning(
            "GGUF not produced. Using merged HF model directory instead. "
            "The container will need transformers-based inference."
        )
        return merged_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Build a customer's agent Docker image")
    parser.add_argument("--business-name", required=True)
    parser.add_argument("--business-type", default="general")
    parser.add_argument("--data",          required=True, help="Directory of customer documents")
    parser.add_argument("--tag",           default="babyyoday-agent:latest")
    parser.add_argument("--output-dir",    default=None)

    # Model options — provide ONE of these:
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model-path",
        help="Path to an existing .gguf model (skip fine-tuning)",
    )
    model_group.add_argument(
        "--lora-config",
        help="Path to a LoRA training config YAML — triggers fine-tuning",
    )

    parser.add_argument(
        "--base-model",
        default=None,
        help="HuggingFace model ID (used with --lora-config, overrides config value)",
    )

    args = parser.parse_args()

    staging = Path(args.output_dir or "/tmp/babyyoday_build")
    staging.mkdir(parents=True, exist_ok=True)

    data_staging  = staging / "data"
    docs_staging  = data_staging / "docs"
    models_staging = staging / "models"

    for d in [docs_staging, data_staging / "incoming", models_staging]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Copy customer documents ────────────────────────────────────────────
    logger.info("Copying documents from %s ...", args.data)
    for f in Path(args.data).iterdir():
        if f.is_file():
            shutil.copy2(f, docs_staging / f.name)

    # ── 2. Build FAISS index ──────────────────────────────────────────────────
    logger.info("Building RAG index ...")
    reindex(docs_dir=str(docs_staging), output_dir=str(data_staging))

    # ── 3. Resolve model ─────────────────────────────────────────────────────
    model_file: Path | None = None

    if args.lora_config:
        logger.info("Fine-tuning mode selected — this will take a while ...")
        model_result = run_fine_tuning(
            business_name=args.business_name,
            docs_dir=docs_staging,
            base_model=args.base_model,
            lora_config_path=args.lora_config,
            staging=staging,
        )
        model_file = model_result if model_result.suffix == ".gguf" else None
        if model_file is None:
            logger.warning(
                "Fine-tuned model is not in GGUF format. "
                "The container will run in retrieval-only mode."
            )

    elif args.model_path:
        src = Path(args.model_path)
        if src.exists():
            dest = models_staging / "model.gguf"
            shutil.copy2(src, dest)
            model_file = dest
            logger.info("Model copied: %s", src.name)
        else:
            logger.warning(
                "Model not found at %s — container will run in retrieval-only mode",
                args.model_path,
            )

    else:
        logger.info(
            "No model specified — container will run in retrieval-only mode. "
            "Add --model-path or --lora-config to get full LLM answers."
        )

    # ── 4. Generate config ────────────────────────────────────────────────────
    config = build_config(args.business_name, args.business_type)
    with open(staging / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # ── 5. Copy application code ──────────────────────────────────────────────
    app_staging = staging / "app_code"
    for module in ["inference", "agent", "admin", "data_pipeline"]:
        src_module = PROJECT_ROOT / module
        dst_module = app_staging / module
        if src_module.exists():
            shutil.copytree(src_module, dst_module, dirs_exist_ok=True)

    shutil.copy2(PROJECT_ROOT / "inference" / "requirements.txt", staging / "requirements.txt")
    shutil.copy2(BUILDER_DIR / "Dockerfile", staging / "Dockerfile")

    # ── 6. Docker build ───────────────────────────────────────────────────────
    logger.info("Building Docker image: %s ...", args.tag)
    result = subprocess.run(
        ["docker", "build", "-t", args.tag, "."],
        cwd=str(staging),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Docker build failed:\n%s", result.stderr)
        sys.exit(1)

    logger.info("Image built: %s", args.tag)
    fine_tuned = "fine-tuned SLM" if args.lora_config else ("pre-trained SLM" if model_file else "retrieval-only")
    logger.info("Mode: %s", fine_tuned)
    logger.info("Run with: docker run -p 8000:8000 -p 8001:8001 %s", args.tag)


if __name__ == "__main__":
    main()
