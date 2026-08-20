# Echo

Voice RAG for HH GOA — ask in speech or text, get a grounded span back from a slice of [MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI). If the index cannot support an answer, Echo refuses instead of inventing one.

```
mic / text → Sarvam STT (optional) → retrieve (BM25 + FAISS) → extract → answer or refuse
```

No LLM on the answer path. Retrieval is hybrid over ~7.8M passages. The index lives as local files in `vaani_index/` (~12 GB: sqlite + FAISS + per-language BM25), not a hosted vector DB.

---

## What we built

A full ask loop, not a notebook demo.

**Pipeline.** Every turn is a tool harness: safety check → language + query-type classify → hybrid retrieve → extractive span → grounding check. Each tool has typed I/O, one retry on retrieve/extract, and a recorded trace. Refusal is a first-class outcome, not an exception.

**Retrieval.** BM25 (bm25s, numba, mmap) is sharded by language so a query scans ~550K docs instead of the full 7.8M. Dense search is `paraphrase-multilingual-MiniLM-L12-v2` over a FAISS IVFPQ index (384-d, ONNX O3 on CPU). Lists fuse with RRF, then a lexical reranker. Dense only runs when BM25 does not already look strong — that skip is most of the latency win.

**Languages.** Fourteen BM25 shards, routed by script + cheap lexical cues so Devanagari (hi / mr / ne / sa) and Eastern Nagari (bn / as) do not search every sibling:

`en` · `hi` · `mr` · `bn` · `gu` · `ta` · `kn` · `ml` · `pa` · `or` · `ur` · `as` · `ne` · `sa`

**Answering.** Sub-10 ms extractive reader. Candidate sentences and clauses are scored with lexical overlap, phrase hits, and type-aware bonuses (LOCATION / PERSON / ENTITY / NUMERIC / DESCRIPTION). No model call. Answers that fail the 55% grounding coverage check are dropped; the next span is tried, then we refuse.

**Guardrails.** Input filters (weapons, violence, self-harm, CSEA, crime, jailbreak, off-topic). Retrieval must beat a score + query-support floor. Output must be covered by the retrieved parents. Thin, empty, or too-long queries are refused.

**Voice.** Optional. Browser records WebM; the API converts to WAV and calls Sarvam Saaras v3. Several API keys rotate on 401 / 402 / 403 / 429 / 5xx. Typed questions work with no keys.

**UI.** React + TypeScript. Home is the composer (text + mic). Result shows the answer, citations, harness trace, and an analytics panel with P50 / P70 / P100 gauges plus per-stage bars. Theme is HH GOA.

**API.** FastAPI on `:8080`.

| Method | Path | What |
| --- | --- | --- |
| `GET` | `/api/health` | index ready, chunk count, Sarvam keys, SLA |
| `POST` | `/api/ask` | `{ "query": "…" }` → extractive result |
| `POST` | `/api/transcribe` | audio file → text + STT ms |
| `POST` | `/api/ask-audio` | audio file → full pipeline |
| `GET` | `/api/metrics` | live latencies + startup sweep (P50 / P70 / P100) |

---

## Latency

SLA is **< 200 ms** for retrieve + extract (STT is out of band — it is a network call to Sarvam).

On boot the API warms the index, then runs a **120-query sweep** (warmup + an unrecorded prime pass, then a recorded pass) so the gauges are never a single ask and never a cold page-fault tail. Watch the server log for:

```
latency sweep ready: n=120 p50=… p70=… p100=…
```

The Result page polls `/api/metrics` and shows those three percentiles plus `% under 200 ms`.

What actually made the budget:

| Problem | What we did |
| --- | --- |
| Full-corpus BM25 ~900 ms | Shard by language (~1/14th the array). Numba + mmap. |
| 100 ms+ tail on Indic queries | Route to **one** shard. Sibling hi→mr/sa/ne/en searches were the tail. |
| ONNX MiniLM saturates the CPU | Never encode in parallel with BM25. Encode only if dense is needed. |
| Dense on every query | Skip FAISS when BM25 already covers the specific terms. |
| Torch MiniLM on CPU | ONNX O3 graph, 1 intra/inter thread, `max_seq_length=64`. ~2× vs torch. |
| First query is a JIT + page-fault hit | Warm BM25 shards, encode a dummy query, sweep 120 queries at startup. |
| Generating an answer with an LLM | Extractive span scorer. No model on the SLA path. |
| sqlite cold reads on 8 GB of passages | `mmap_size=1G`, 128 MB page cache, 50k chunk LRU, query-vector LRU. |

`benchmark.py` is the offline suite (pure FAISS, end-to-end RAG, 14-language breakdown, token-F1 vs gold). The live product uses the startup sweep above.

```bash
PYTHONPATH=src python benchmark.py                 # loads the index in-process
PYTHONPATH=src python benchmark.py --http http://127.0.0.1:8080
PYTHONPATH=src python benchmark.py -n 200 --json data/reports/benchmark.json
```

---

## Layout

```
src/voice_rag/     pipeline, hybrid retrieval, harness, guardrails, Sarvam, latency
frontend/          React + TypeScript UI (Vite)
scripts/           serve.py, build_bm25.py
benchmark.py       offline P50 / quality / multilingual suite
tests/             harness, retrieval, guardrails, latency sweep, STT
vaani_index/       passages.db + lsa.faiss + bm25/<lang>/   (~12 GB, not in git)
```

---

## Setup

### Prerequisites

- Python **3.11+** (3.12 is what we run)
- Node **20+** (22 in Docker)
- ~16 GB RAM to load the index comfortably
- `vaani_index/` already built and sitting next to this README (`encoder.meta.json` + `passages.db` + `bm25/` + `lsa.faiss`)
- A [Sarvam](https://sarvam.ai) API key **only if you want the mic**

The 12 GB index is gitignored. Get it from whoever already built it — there is no ingest script in this repo.

### 1. Python API

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy the env template and add a key if you have one:

```bash
cp .env.example .env
# SARVAM_API_KEY=sk_...
# optional extras (rotation): SARVAM_API_KEY_2=…  SARVAM_API_KEYS=k1,k2
```

Start the API:

```bash
PYTHONPATH=src python scripts/serve.py
```

Wait until you see `latency sweep ready: …`. Health check: [http://127.0.0.1:8080/api/health](http://127.0.0.1:8080/api/health).

Typed ask with no UI:

```bash
curl -s http://127.0.0.1:8080/api/ask \
  -H 'content-type: application/json' \
  -d '{"query":"who invented the telephone"}'
```

### 2. React UI (dev)

Second terminal:

```bash
cd frontend
npm install
npm run dev
```

UI: [http://127.0.0.1:5173](http://127.0.0.1:5173) · Vite proxies `/api` to `:8080`.

### 3. Production (one process)

```bash
cd frontend && npm run build && cd ..
PYTHONPATH=src python scripts/serve.py
```

FastAPI serves `frontend/dist` on [http://127.0.0.1:8080](http://127.0.0.1:8080).

### Tests

```bash
source .venv/bin/activate
PYTHONPATH=src pytest -q
```

---

## Docker / AWS

`vaani_index/` is **12 GB** and is not in the image. Mount it. Use at least a **t3.large** (8 GB RAM) and **30 GB** disk. `t3.medium` / 4 GB will OOM. Compose caps the app at 6 GB.

From your laptop (replace `IP` and the key):

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
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2
sudo usermod -aG docker ubuntu
# log out/in, or: newgrp docker

cd ~/echo
test -f vaani_index/encoder.meta.json && test -f vaani_index/passages.db
# .env should exist so SARVAM_API_KEY is set if you want the mic

docker compose up -d --build
docker compose logs -f
```

Wait for `latency sweep ready: n=120 p50=… p70=… p100=…`. Then open `http://IP:8080`.

Caddy is in the compose file and terminates TLS on 80/443 for `${SITE_HOST}` (default `16.171.133.1.sslip.io`). **The mic needs HTTPS** — browsers hide `getUserMedia` on a raw `http://IP` page. Use the sslip.io URL, or type the question.

Security group: inbound **22** (SSH), **80**, **443**, and **8080** from your IP (or `0.0.0.0/0` for the demo).
