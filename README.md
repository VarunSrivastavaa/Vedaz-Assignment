Vedaz AI Astrologer — Technical Assessment
This repository contains my submission for the AI/ML Engineer technical assessment: fine-tuning a Qwen model on astrology chat data, a guide for hosting the model on a VPS with vLLM, and sample training conversations.

📁 Repository Structure

├── finetune/
│   ├── train.py            # LoRA/QLoRA fine-tuning script (Unsloth)
│   ├── merge_lora.py       # Merges trained adapter into base model
│   ├── requirements.txt    # Python dependencies
│   └── README.md           # Fine-tuning setup & usage instructions
├── data/
│   └── chat_data.jsonl     # Provided training dataset (175 conversations)
├── writeup1_vllm_hosting.md   # VPS hosting guide using vLLM
├── writeup2_sample_chats.jsonl # 5 manually written training conversations
└── README.md                # This file

🧠 Part 1: Fine-tuning
The provided dataset (data/chat_data.jsonl) contains 175 conversations in OpenAI-style messages format (system/user/assistant), covering astrology guidance across relationships, career, health, and finance — with built-in safety behavior (e.g. no fatalistic predictions, crisis redirection to helplines).
Approach: LoRA (4-bit QLoRA) fine-tuning of Qwen2.5-7B-Instruct using Unsloth, chosen over full fine-tuning to avoid overfitting on a small dataset and to preserve the base model's general reasoning and safety behavior while adapting its tone and domain knowledge.
See finetune/README.md for full setup and run instructions.

🖥️ Part 2: Hosting with vLLM
writeup1_vllm_hosting.md walks through provisioning a GPU VPS, installing vLLM, serving the fine-tuned model via an OpenAI-compatible API, and securing it with Nginx + systemd for production use.

💬 Part 3: Sample Training Conversations
writeup2_sample_chats.jsonl contains 5 manually written astrologer-user conversations, each demonstrating:

Kundli/Vedic astrology knowledge (grahas, bhavas, dashas)
A "please wait a minute while I analyze your kundli" step
Empathetic, non-judgmental responses
Time-bound but non-fatalistic future predictions

⚙️ Tech Stack

Model: Qwen2.5-7B-Instruct (Qwen3 also supported)
Fine-tuning: Unsloth, PEFT (LoRA), TRL, bitsandbytes
Serving: vLLM (OpenAI-compatible API)
Data format: JSONL, messages schema
