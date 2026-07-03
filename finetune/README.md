# Vedaz AI Astrologer — Qwen2.5/Qwen3 Fine-tuning

Yeh folder Qwen2.5-7B-Instruct (ya Qwen3-8B) ko diye gaye astrology-chat dataset
par LoRA fine-tune karne ke liye code contain karta hai.

## Kya use kiya gaya hai
- **Base model**: `Qwen/Qwen2.5-7B-Instruct` (ya `Qwen/Qwen3-8B` — config mein switch kar sakte ho)
- **Method**: LoRA (4-bit QLoRA) via [Unsloth](https://github.com/unslothai/unsloth) — fast aur low-VRAM
- **Data**: `data/chat_data.jsonl` — 175 records, har record `{"messages": [system, user, assistant, ...]}` format mein

## Requirements
- GPU with at least 16GB VRAM (T4/A10/A100 sab chalega, 4-bit quantization use ho rahi hai)
- Python 3.10+, CUDA 12.x

## Setup

```bash
pip install unsloth
pip install --no-deps trl peft accelerate bitsandbytes
```

## Run fine-tuning

```bash
python train.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --data_path ../data/chat_data.jsonl \
  --output_dir ./output/vedaz-astrologer-lora \
  --epochs 3 \
  --lr 2e-4 \
  --max_seq_len 2048
```

Isse ek LoRA adapter train hoga (base model modify nahi hota, sirf chhote adapter
weights save hote hain — ~50-200MB).

## Merge adapter into base model (optional, deployment ke liye)

```bash
python merge_lora.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path ./output/vedaz-astrologer-lora \
  --merged_output ./output/vedaz-astrologer-merged
```

Merged model ko phir vLLM se serve kar sakte ho (dekho `../writeup1_vllm_hosting.md`).

## Data format sample

```json
{"messages": [
  {"role": "system", "content": "You are Vedaz's AI Vedic astrologer..."},
  {"role": "user", "content": "Mera breakup ho gaya hai..."},
  {"role": "assistant", "content": "यह सुनकर मुझे बहुत चिंता हो रही है..."}
]}
```

## Notes on why this approach
- **LoRA over full fine-tune**: dataset chhota hai (175 examples), full fine-tune
  se overfitting aur catastrophic forgetting ka risk hai. LoRA base model ki
  general capability retain karte hue sirf tone/domain-behavior adjust karta hai.
- **Safety behavior preservation**: dataset mein crisis-handling examples
  (self-harm, child illness) explicitly include hain taaki fine-tuning ke baad
  bhi model helpline-redirect behavior na bhoole — is wajah se low learning
  rate aur kam epochs (3) rakhe gaye hain, taaki over-fit na ho.
