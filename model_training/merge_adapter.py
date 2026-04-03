#!/usr/bin/env python3
"""
Merge a LoRA adapter into the base model and export to GGUF.

Steps:
  1. Load base model + LoRA adapter
  2. Merge adapter weights into base model
  3. Save merged model as HuggingFace checkpoint
  4. Convert to GGUF using llama.cpp's convert script (if available)
     or print instructions for manual conversion

The final .gguf file is what gets baked into the customer's Docker container.

Usage:
    python model_training/merge_adapter.py \
        --adapter-path ./models/lora_adapter/ \
        --output-dir   ./models/merged/ \
        --gguf-output  ./models/model.gguf \
        --quantization q4_k_m
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

QUANTIZATION_OPTIONS = ["q4_k_m", "q5_k_m", "q8_0", "f16"]
DEFAULT_QUANTIZATION = "q4_k_m"  # 4-bit, good balance of size/quality


def load_adapter_meta(adapter_path: str) -> dict:
    meta_path = Path(adapter_path) / "adapter_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    logger.warning("adapter_meta.json not found — you may need to specify --base-model")
    return {}


def merge(adapter_path: str, output_dir: str, base_model: str | None = None):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        logger.error("Missing dependency: %s\nInstall: pip install transformers peft", e)
        sys.exit(1)

    meta = load_adapter_meta(adapter_path)
    resolved_base = base_model or meta.get("base_model")
    if not resolved_base:
        logger.error(
            "Cannot determine base model. Pass --base-model explicitly."
        )
        sys.exit(1)

    logger.info("Loading base model: %s", resolved_base)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_base,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(resolved_base, trust_remote_code=True)

    logger.info("Loading LoRA adapter from: %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)

    logger.info("Merging adapter weights ...")
    model = model.merge_and_unload()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Saving merged model to: %s", out)
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))

    logger.info("Merged model saved. Size: %.1f GB", _dir_size_gb(out))
    return out


def _dir_size_gb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 ** 3)


def convert_to_gguf(
    merged_model_dir: str,
    gguf_output: str,
    quantization: str = DEFAULT_QUANTIZATION,
    llama_cpp_dir: str | None = None,
) -> bool:
    """
    Convert a merged HuggingFace model to GGUF using llama.cpp.
    Returns True if successful, False if llama.cpp is not available.
    """
    gguf_path = Path(gguf_output)
    gguf_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to find llama.cpp convert script
    candidates = [
        llama_cpp_dir and Path(llama_cpp_dir) / "convert_hf_to_gguf.py",
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        Path("/opt/llama.cpp/convert_hf_to_gguf.py"),
        Path("./llama.cpp/convert_hf_to_gguf.py"),
    ]
    convert_script = next((c for c in candidates if c and c.exists()), None)

    if convert_script is None:
        logger.warning(
            "\nllama.cpp convert script not found.\n"
            "To convert to GGUF manually:\n"
            "  1. Clone llama.cpp:  git clone https://github.com/ggerganov/llama.cpp\n"
            "  2. Run:\n"
            "     python llama.cpp/convert_hf_to_gguf.py %s \\\n"
            "       --outfile %s \\\n"
            "       --outtype %s\n"
            "Or use the HuggingFace Space: https://huggingface.co/spaces/ggml-org/gguf-my-repo",
            merged_model_dir,
            gguf_output,
            quantization,
        )
        return False

    fp16_path = str(gguf_path).replace(".gguf", "_fp16.gguf")

    # Step 1: Convert to fp16 GGUF
    logger.info("Converting to fp16 GGUF ...")
    result = subprocess.run(
        [
            sys.executable,
            str(convert_script),
            merged_model_dir,
            "--outfile", fp16_path,
            "--outtype", "f16",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error("Conversion failed:\n%s", result.stderr)
        return False

    # Step 2: Quantize to target format
    logger.info("Quantizing to %s ...", quantization)
    quantize_bin = Path(convert_script).parent / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = Path(convert_script).parent / "quantize"  # older name

    if quantize_bin.exists():
        result = subprocess.run(
            [str(quantize_bin), fp16_path, gguf_output, quantization.upper()],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.warning("Quantization failed — using fp16: %s", result.stderr)
            Path(fp16_path).rename(gguf_path)
        else:
            Path(fp16_path).unlink(missing_ok=True)
            logger.info("GGUF saved: %s (%.1f GB)", gguf_path, gguf_path.stat().st_size / 1e9)
    else:
        logger.info("quantize binary not found — keeping fp16 GGUF at %s", fp16_path)
        Path(fp16_path).rename(gguf_path)

    return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Merge LoRA adapter and export to GGUF")
    parser.add_argument("--adapter-path",  required=True, help="Path to saved LoRA adapter")
    parser.add_argument("--output-dir",    required=True, help="Where to save the merged HF model")
    parser.add_argument("--gguf-output",   default="./models/model.gguf",
                        help="Final GGUF path (baked into container)")
    parser.add_argument("--base-model",    default=None,
                        help="Override base model (otherwise read from adapter_meta.json)")
    parser.add_argument("--quantization",  default=DEFAULT_QUANTIZATION,
                        choices=QUANTIZATION_OPTIONS)
    parser.add_argument("--llama-cpp-dir", default=None,
                        help="Path to llama.cpp source directory")
    parser.add_argument("--skip-gguf",     action="store_true",
                        help="Only merge, don't convert to GGUF")
    args = parser.parse_args()

    merged_dir = merge(args.adapter_path, args.output_dir, args.base_model)

    if args.skip_gguf:
        logger.info("Skipping GGUF conversion (--skip-gguf set)")
        logger.info("Merged model at: %s", merged_dir)
        return

    success = convert_to_gguf(
        merged_model_dir=str(merged_dir),
        gguf_output=args.gguf_output,
        quantization=args.quantization,
        llama_cpp_dir=args.llama_cpp_dir,
    )

    if success:
        logger.info("\nDone! GGUF model ready at: %s", args.gguf_output)
        logger.info("Rebuild the container:")
        logger.info(
            "  python builder/build_customer.py "
            "--model-path %s <other args>",
            args.gguf_output,
        )
    else:
        logger.info("\nMerged HF model at: %s", merged_dir)
        logger.info("Convert to GGUF manually, then rebuild the container.")


if __name__ == "__main__":
    main()
