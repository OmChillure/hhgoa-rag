from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from voice_rag.config import ROOT, settings
from voice_rag.latency.metrics import summarize
from voice_rag.pipeline import VoiceRAG
from voice_rag.stt.sarvam import SarvamError, transcribe
from voice_rag.types import PipelineResult, StageTiming

rag = VoiceRAG()
_live_latencies: list[float] = []


@asynccontextmanager
async def lifespan(_: FastAPI):
    idx = settings.index_dir
    if (idx / "encoder.meta.json").exists() or (idx / "READY").exists():
        rag.load(idx)
    yield


app = FastAPI(title="Vaani", version="0.1.0", lifespan=lifespan, default_response_class=ORJSONResponse)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskBody(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    stt_ms: float | None = None


def _attach_stt(result: PipelineResult, stt_ms: float | None) -> PipelineResult:
    if stt_ms is None or stt_ms < 0:
        return result
    result.stt_ms = float(stt_ms)
    if not any(t.name == "stt" for t in result.timings):
        result.timings = [StageTiming(name="stt", ms=result.stt_ms), *result.timings]
    return result


def _serialize(result: PipelineResult) -> dict[str, Any]:
    _live_latencies.append(result.total_ms)
    if len(_live_latencies) > 500:
        del _live_latencies[: len(_live_latencies) - 500]
    return result.model_dump()


@app.get("/api/health")
def health() -> dict[str, Any]:
    ready = rag.harness is not None
    stats = rag.retriever.stats() if ready else {}
    return {
        "ok": True,
        "ready": ready,
        "sarvam": bool(settings.sarvam_key_list()),
        "sarvam_keys": len(settings.sarvam_key_list()),
        "stats": stats,
        "sla_ms": settings.sla_ms,
    }


@app.post("/api/ask")
def ask(body: AskBody) -> dict[str, Any]:
    if rag.harness is None:
        raise HTTPException(503, "Index not built. Run: python scripts/ingest.py")
    return _serialize(_attach_stt(rag.ask(body.query), body.stt_ms))


@app.post("/api/transcribe")
async def transcribe_only(file: UploadFile = File(...)) -> dict[str, Any]:
    audio = await file.read()
    try:
        tr = transcribe(
            audio,
            filename=file.filename or "audio.webm",
            content_type=file.content_type or "audio/webm",
        )
    except SarvamError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"text": tr.text, "ms": tr.ms, "language_code": tr.language_code}


@app.post("/api/ask-audio")
async def ask_audio(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    if rag.harness is None:
        raise HTTPException(503, "Index not built. Run: python scripts/ingest.py")
    audio = await file.read()
    try:
        result = rag.ask_audio(
            audio,
            filename=file.filename or "audio.webm",
            content_type=file.content_type or "audio/webm",
        )
    except SarvamError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _serialize(result)


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    bench_path = settings.reports_dir / "latency.json"
    bench = {}
    if bench_path.exists():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
    return {
        "live": summarize(_live_latencies),
        "bench": bench,
    }


@app.get("/api/samples")
def samples() -> dict[str, Any]:
    path = settings.index_dir / "holdout_queries.json"
    if not path.exists():
        return {"queries": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for q in data[:12]:
        out.append(
            {
                "query": q.get("en_query"),
                "hi_query": q.get("hi_query"),
                "query_type": q.get("query_type"),
            }
        )
    return {"queries": out}


DIST = ROOT / "frontend" / "dist"


@app.get("/")
def index() -> FileResponse:
    page = DIST / "index.html"
    if not page.exists():
        raise HTTPException(404, "UI not built. cd frontend && npm run build")
    return FileResponse(page)


if (DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/{full_path:path}")
def spa(full_path: str) -> FileResponse:
    if full_path.startswith("api/"):
        raise HTTPException(404, "not found")
    if not DIST.exists():
        raise HTTPException(404, "UI not built. cd frontend && npm run build")
    candidate = DIST / full_path
    if candidate.is_file():
        return FileResponse(candidate)
    return FileResponse(DIST / "index.html")
