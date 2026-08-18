# Echo

Voice RAG over a slice of [MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI).

```
mic / text → Sarvam STT (optional) → retrieve (BM25 + FAISS) → extract → answer or refuse
```

Index is local files in `vaani_index/` (~12 GB: sqlite + FAISS + BM25). Not a hosted vector DB.

## Run

```bash
cd /media/omchillure/Hackathon/rag
source .venv/bin/activate
# terminal 1 — API
PYTHONPATH=src python scripts/serve.py
# terminal 2 — React UI (HH GOA theme)
cd frontend && npm install && npm run dev
```

UI: http://127.0.0.1:5173  ·  API: http://127.0.0.1:8080

Production (API serves the built UI):

```bash
cd frontend && npm run build
PYTHONPATH=src python scripts/serve.py
```

Typed questions work with no keys. Mic needs Sarvam. Answers are extractive (no LLM).

## One-time index (already built)

```bash
# data/raw/hinval.parquet must exist (~441 MB Hindi validation split)
PYTHONPATH=src python scripts/ingest.py
PYTHONPATH=src python scripts/bench.py
```

## AWS (EC2 + Docker)

`vaani_index/` is **12 GB** and is gitignored. Ship it to the VM yourself. Use at least a **t3.large** (8 GB RAM) and **30 GB** disk. `t3.medium` / 4 GB will OOM.

On the laptop (replace `IP` and the key):

```bash
# 1. repo (no venv, no node_modules)
rsync -avz --progress \
  --exclude '.venv' --exclude 'frontend/node_modules' --exclude '.git' \
  -e "ssh -i ~/.ssh/your-key.pem" \
  ./ ubuntu@IP:~/echo/

# 2. index (this is the slow one, ~12 GB)
rsync -avz --progress \
  -e "ssh -i ~/.ssh/your-key.pem" \
  ./vaani_index/ ubuntu@IP:~/echo/vaani_index/
```

On the VM:

```bash
# Docker + compose
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2
sudo usermod -aG docker ubuntu
# log out/in, or: newgrp docker

cd ~/echo
test -f vaani_index/encoder.meta.json && test -f vaani_index/passages.db
touch .env   # or copy yours so SARVAM_API_KEY is set (mic only)

docker compose up -d --build
docker compose logs -f
```

Wait for `latency sweep ready: n=120 p50=… p70=… p100=…`. Then open `http://IP:8080`.

Security group: inbound **22** (SSH) and **8080** (app) from your IP (or `0.0.0.0/0` for the demo).

## Layout

```
src/voice_rag/   pipeline, retrieval, harness, guardrails, Sarvam
frontend/        React + TypeScript UI
scripts/         serve.py
vaani_index/     FAISS + BM25 + passages.db  (upload separately)
```
