#!/usr/bin/env python3
"""
LoRA fine-tuning for the SLM using QLoRA (4-bit) + PEFT + TRL.

What this does:
  - Loads a base model (Phi-3-mini, Mistral-7B, Llama-3-8B, etc.)
  - Applies 4-bit quantization so it fits on small hardware (Mac M-series, CPU, low VRAM)
  - Fine-tunes with LoRA on the business's Q&A training data
  - Saves the LoRA adapter (~20-80 MB, not the full model)
  - The adapter is later merged + exported to GGUF for the container (see merge_adapter.py)

Requirements:
    pip install transformers peft accelerate bitsandbytes datasets trl

Usage:
    python model_training/train_lora.py \
        --config model_training/configs/phi3_lora.yaml

    python model_training/train_lora.py \
        --base-model microsoft/Phi-3-mini-4k-instruct \
        --training-data ./data/training_data.jsonl \
        --output ./models/phi3_lora_adapter/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_training_data(jsonl_path: str) -> list[dict]:
    data = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    logger.info("Loaded %d training examples from %s", len(data), jsonl_path)
    return data


def format_chatml(example: dict) -> str:
    """Convert a training example dict into ChatML string for fine-tuning."""
    return (
        f"<|im_start|>system\n{example['system']}<|im_end|>\n"
        f"<|im_start|>user\n{example['user']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['assistant']}<|im_end|>"
    )


def format_phi3(example: dict) -> str:
    return (
        f"<|system|>\n{example['system']}<|end|>\n"
        f"<|user|>\n{example['user']}<|end|>\n"
        f"<|assistant|>\n{example['assistant']}<|end|>"
    )


def format_mistral(example: dict) -> str:
    return (
        f"[INST] <<SYS>>\n{example['system']}\n<</SYS>>\n\n"
        f"{example['user']} [/INST] {example['assistant']}"
    )


def format_llama3(example: dict) -> str:
    return (
        "<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{example['system']}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{example['user']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n{example['assistant']}<|eot_id|>"
    )


FORMATTERS = {
    "phi3":    format_phi3,
    "mistral": format_mistral,
    "llama3":  format_llama3,
    "chatml":  format_chatml,
}


def detect_model_family(model_name_or_path: str) -> str:
    name = model_name_or_path.lower()
    if "phi" in name:
        return "phi3"
    if "mistral" in name or "mixtral" in name:
        return "mistral"
    if "llama-3" in name or "llama3" in name:
        return "llama3"
    return "chatml"


def train(cfg: dict):
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer
    except ImportError as e:
        logger.error(
            "Missing training dependency: %s\n"
            "Install with: pip install transformers peft accelerate bitsandbytes datasets trl",
            e,
        )
        sys.exit(1)

    base_model = cfg["base_model"]
    training_data_path = cfg["training_data"]
    output_dir = cfg["output_dir"]
    model_family = cfg.get("model_family") or detect_model_family(base_model)

    lora_cfg = cfg.get("lora", {})
    train_cfg = cfg.get("training", {})

    logger.info("Base model    : %s", base_model)
    logger.info("Model family  : %s", model_family)
    logger.info("Training data : %s", training_data_path)
    logger.info("Output dir    : %s", output_dir)

    # ── 1. Load tokenizer ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 2. Quantization config (QLoRA = LoRA on 4-bit model) ─────────────────
    use_4bit = cfg.get("quantization", {}).get("load_in_4bit", True)
    bnb_config = None
    if use_4bit:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            logger.info("Using 4-bit QLoRA")
        except Exception:
            logger.warning("bitsandbytes 4-bit not available — loading in fp16")
            bnb_config = None

    # ── 3. Load base model ───────────────────────────────────────────────────
    device_map = "auto" if torch.cuda.is_available() else (
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Device map: %s", device_map)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16 if bnb_config is None else None,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # ── 4. LoRA config ───────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 5. Prepare dataset ───────────────────────────────────────────────────
    raw_data = load_training_data(training_data_path)
    formatter = FORMATTERS.get(model_family, format_chatml)
    texts = [formatter(ex) for ex in raw_data]
    dataset = Dataset.from_dict({"text": texts})

    split = train_cfg.get("val_split", 0.1)
    if split > 0 and len(dataset) > 10:
        splits = dataset.train_test_split(test_size=split, seed=42)
        train_dataset = splits["train"]
        eval_dataset = splits["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    logger.info(
        "Train: %d examples | Eval: %d examples",
        len(train_dataset),
        len(eval_dataset) if eval_dataset else 0,
    )

    # ── 6. Training arguments ────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("epochs", 3),
        per_device_train_batch_size=train_cfg.get("batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 0.001),
        fp16=torch.cuda.is_available(),
        bf16=False,
        max_grad_norm=train_cfg.get("max_grad_norm", 0.3),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 50),
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=train_cfg.get("eval_steps", 50) if eval_dataset else None,
        load_best_model_at_end=bool(eval_dataset),
        report_to="none",
        push_to_hub=False,
    )

    # ── 7. SFT Trainer ───────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=train_cfg.get("max_seq_length", 1024),
        packing=train_cfg.get("packing", False),
    )

    logger.info("Starting training ...")
    trainer.train()

    # ── 8. Save adapter ──────────────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    # Save metadata so merge_adapter knows what base model was used
    meta = {
        "base_model": base_model,
        "model_family": model_family,
        "lora_r": lora_cfg.get("r", 16),
    }
    with open(out / "adapter_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("LoRA adapter saved to %s", out)
    logger.info("Next: run merge_adapter.py to merge + export to GGUF")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Fine-tune SLM with LoRA")
    parser.add_argument("--config",        default=None,  help="YAML config file")
    parser.add_argument("--base-model",    default=None)
    parser.add_argument("--training-data", default=None)
    parser.add_argument("--output",        default="./models/lora_adapter/")
    parser.add_argument("--epochs",        type=int, default=None)
    parser.add_argument("--batch-size",    type=int, default=None)
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
    elif args.base_model and args.training_data:
        cfg = {
            "base_model": args.base_model,
            "training_data": args.training_data,
            "output_dir": args.output,
        }
    else:
        parser.error("Provide --config or both --base-model and --training-data")

    # CLI overrides
    if args.epochs:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size
    if args.output != "./models/lora_adapter/":
        cfg["output_dir"] = args.output

    train(cfg)


if __name__ == "__main__":
    main()
