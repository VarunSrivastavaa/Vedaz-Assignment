"""
Fine-tune Qwen2.5 / Qwen3 on the Vedaz astrologer chat dataset using
Unsloth (LoRA / QLoRA). Run on a machine with a CUDA GPU.

Usage:
    python train.py --base_model Qwen/Qwen2.5-7B-Instruct \
                     --data_path ../data/chat_data.jsonl \
                     --output_dir ./output/vedaz-astrologer-lora
"""

import argparse
import json

from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./output/vedaz-astrologer-lora")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    return p.parse_args()


def load_jsonl_dataset(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def main():
    args = parse_args()

    # 1. Load base model in 4-bit (QLoRA) via Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        dtype=None,          # auto-detect (bf16/fp16)
        load_in_4bit=True,
    )

    # 2. Attach LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # 3. Apply Qwen chat template so `messages` -> prompt string correctly
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    def formatting_func(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    raw_ds = load_jsonl_dataset(args.data_path)
    ds = raw_ds.map(formatting_func, remove_columns=raw_ds.column_names)

    # 4. Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_ratio=0.05,
            logging_steps=5,
            save_strategy="epoch",
            output_dir=args.output_dir,
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            bf16=True,
            report_to="none",
        ),
    )

    trainer.train()

    # 5. Save LoRA adapter
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
