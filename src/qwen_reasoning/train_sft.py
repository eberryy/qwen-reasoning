"""Executable core of the completed SFT configuration; private gates omitted."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3-4B-Base"
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
TASKS = {"bit_manipulation", "encryption", "equations", "gravity", "numeral_system", "unit_conversion"}


def load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected object")
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id or row_id in seen:
                raise ValueError(f"line {line_number}: missing or duplicate ID")
            if row.get("task_type") not in TASKS:
                raise ValueError(f"line {line_number}: unknown task type")
            prompt, completion = row.get("prompt"), row.get("completion")
            if not isinstance(prompt, list) or not prompt or not isinstance(prompt[-1], dict) or prompt[-1].get("role") != "user":
                raise ValueError(f"line {line_number}: prompt must end with user")
            if not isinstance(completion, list) or len(completion) != 1 or not isinstance(completion[0], dict) or completion[0].get("role") != "assistant":
                raise ValueError(f"line {line_number}: expected one assistant completion")
            for message in prompt + completion:
                if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"} or not isinstance(message.get("content"), str) or not message["content"].strip():
                    raise ValueError(f"line {line_number}: invalid chat message")
            seen.add(row_id)
            rows.append(row)
    if not rows:
        raise ValueError("training input is empty")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    rows = load_records(args.dataset)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    print(f"Loaded {len(rows)} rows: {dict(sorted(Counter(r['task_type'] for r in rows).items()))}")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("SFT requires a CUDA GPU")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.padding_side = "right"
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos_id is None or eos_id == tokenizer.unk_token_id:
        raise RuntimeError("Model tokenizer lacks configured EOS token")
    tokenizer.eos_token = "<|im_end|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for row in rows:
        tokens = tokenizer.apply_chat_template(
            row["prompt"] + row["completion"],
            tokenize=True,
            add_generation_prompt=False,
        )
        if len(tokens) > 8192:
            raise ValueError(f"Record {row['id']!r} has {len(tokens)} tokens; silent truncation refused")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16,
        device_map={"": local_rank},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    config = SFTConfig(
        output_dir=str(args.output), num_train_epochs=args.epochs,
        learning_rate=2e-4, per_device_train_batch_size=1,
        gradient_accumulation_steps=4, max_length=8192,
        assistant_only_loss=False, completion_only_loss=True,
        eos_token="<|im_end|>", packing=False, bf16=True,
        gradient_checkpointing=True, logging_steps=5,
        save_strategy="steps", save_steps=100, save_total_limit=2,
        report_to="none", seed=42, data_seed=42,
        ddp_find_unused_parameters=False, dataloader_num_workers=4,
    )
    trainer = SFTTrainer(
        model=model, args=config, train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0,
            target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
        ),
    )
    trainer.train()
    trainer.save_model(str(args.output / "final_adapter"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(args.output / "final_adapter"))


if __name__ == "__main__":
    main()
