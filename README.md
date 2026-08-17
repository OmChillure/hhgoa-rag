# Vaani

Voice RAG over a slice of [MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI).

```
mic / text → Sarvam STT (optional) → retrieve (BM25 + FAISS) → extract → answer or refuse
```

Index is local files in `data/index/` (~118 MB). Not a hosted vector DB.

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

Copy this repo **including `data/index/`** to a **t3.medium** (or t3.large if the index grows). Then:

```bash
docker compose up -d --build
```

http://\<public-ip\>:8080 — security group must allow **8080**.

## Layout

```
src/voice_rag/   pipeline, retrieval, harness, guardrails, Sarvam
frontend/        React + TypeScript UI
scripts/         serve.py  ingest.py  bench.py
data/index/      FAISS + BM25 + chunks
```
