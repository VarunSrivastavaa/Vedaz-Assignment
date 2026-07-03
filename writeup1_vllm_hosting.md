# Writeup 1: Fine-tuned Model ko VPS par vLLM Se Host Karna

## Overview

vLLM ek high-throughput inference server hai jo OpenAI-compatible API endpoint
expose karta hai. Isse fine-tuned Qwen model ko production mein serve karna
aasan ho jata hai — koi custom API code likhne ki zaroorat nahi.

## Step 1: VPS Choose Aur Setup Karna

- GPU-enabled VPS lo (jaise RunPod, Lambda Labs, Paperspace, ya AWS/GCP GPU
  instance). 7B model ke liye minimum **24GB VRAM** (A10/A100/RTX 4090) recommend hai.
- Ubuntu 22.04 base image use karo.
- SSH se VPS mein login karo:
  ```bash
  ssh root@<vps-ip>
  ```

## Step 2: System Dependencies Install Karna

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git build-essential
```

NVIDIA driver aur CUDA toolkit already installed hona chahiye (zyada tar GPU
VPS providers pehle se install karke dete hain). Verify karo:

```bash
nvidia-smi
```

## Step 3: Python Environment Banana

```bash
python3 -m venv vllm-env
source vllm-env/bin/activate
pip install --upgrade pip
```

## Step 4: vLLM Install Karna

```bash
pip install vllm
```

## Step 5: Fine-tuned Model VPS Par Copy Karna

Agar model Hugging Face Hub par upload kiya hai:

```bash
huggingface-cli login
huggingface-cli download <your-username>/vedaz-astrologer-merged --local-dir ./model
```

Ya `scp` se local se directly copy kar sakte ho:

```bash
scp -r ./output/vedaz-astrologer-merged root@<vps-ip>:/root/model
```

## Step 6: vLLM Server Start Karna

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /root/model \
  --served-model-name vedaz-astrologer \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

Yeh command ek **OpenAI-compatible `/v1/chat/completions` endpoint** start kar
deta hai `http://<vps-ip>:8000` par.

### Production ke liye background mein chalana (systemd)

```ini
# /etc/systemd/system/vllm.service
[Unit]
Description=vLLM Server
After=network.target

[Service]
User=root
WorkingDirectory=/root
ExecStart=/root/vllm-env/bin/python -m vllm.entrypoints.openai.api_server \
  --model /root/model --served-model-name vedaz-astrologer \
  --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable vllm
sudo systemctl start vllm
```

## Step 7: Firewall Aur Reverse Proxy Setup

Port 8000 seedha public expose na karo — Nginx reverse proxy lagao taaki HTTPS
aur rate-limiting add ho sake:

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Phir Certbot se free SSL certificate laga do (`certbot --nginx`), aur `ufw`
firewall mein sirf 80/443 allow karo, 8000 ko block rakho.

## Step 8: Testing

```bash
curl http://<vps-ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vedaz-astrologer",
    "messages": [
      {"role": "user", "content": "Mera career kaisa rahega is saal?"}
    ]
  }'
```

Response JSON mein model ka generated reply aana chahiye — isse confirm ho
jata hai ki hosting successfully kaam kar rahi hai.

## Notes

- Agar multiple LoRA adapters serve karne hain (base model change kiye bina),
  vLLM `--enable-lora` flag support karta hai — merge karna zaroori nahi.
- Monitoring ke liye vLLM `/metrics` endpoint expose karta hai (Prometheus-compatible).
- Traffic badhne par `--tensor-parallel-size` se multiple GPUs pe scale kar sakte ho.
