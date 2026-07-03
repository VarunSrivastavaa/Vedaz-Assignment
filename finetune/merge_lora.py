"""
Merge a trained LoRA adapter into the base Qwen model so it can be served
directly by vLLM (vLLM prefers a single merged checkpoint for simplicity,
though it can also serve LoRA adapters separately via --enable-lora).

Usage:
    python merge_lora.py --base_model Qwen/Qwen2.5-7B-Instruct \
                          --lora_path ./output/vedaz-astrologer-lora \
                          --merged_output ./output/vedaz-astrologer-merged
"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--lora_path", type=str, required=True)
    p.add_argument("--merged_output", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading LoRA adapter: {args.lora_path}")
    model = PeftModel.from_pretrained(base_model, args.lora_path)

    print("Merging adapter into base weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.merged_output}")
    model.save_pretrained(args.merged_output, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_output)

    print("Done. This directory can now be pointed to directly by vLLM.")


if __name__ == "__main__":
    main()
