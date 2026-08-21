from __future__ import annotations

import json
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from voice_rag.config import ROOT, settings
from voice_rag.latency.bench import load_bench_queries, run_latency_sweep
from voice_rag.latency.metrics import summarize
from voice_rag.pipeline import VoiceRAG
from voice_rag.stt.sarvam import SarvamError, transcribe
from voice_rag.types import PipelineResult, StageTiming

rag = VoiceRAG()
_ask_lock = threading.Lock()
_live_latencies: list[float] = []
_bench: dict[str, Any] = {}
_bench_status = "idle"
_bench_error = ""


def _ask_sync(query: str, **kwargs: Any) -> PipelineResult:
    with _ask_lock:
        return rag.ask(query, **kwargs)


def _run_startup_sweep() -> None:
    """P50/P70/P100 over many index queries. RAM only — no data/reports."""
    global _bench, _bench_status, _bench_error
    if rag.harness is None:
        _bench_status = "error"
        _bench_error = "index not loaded"
        return
    n = max(10, int(settings.bench_n or 120))
    _bench_status = "running"
    print(f"latency sweep: loading up to {n} test queries…", flush=True)
    try:
        queries = load_bench_queries(settings.index_dir, n)
        if not queries:
            _bench_status = "error"
            _bench_error = "no test queries in the index"
            return
        print(f"latency sweep: running {len(queries)} queries…", flush=True)
        payload = run_latency_sweep(
            _ask_sync,
            queries,
            settings.sla_ms,
        )
        if not payload:
            _bench_status = "error"
            _bench_error = "sweep produced no timings"
            return
        _bench = payload
        _bench_status = "ready"
        lat = payload.get("latency") or {}
        print(
            f"latency sweep ready: n={payload.get('n_queries')} "
            f"p50={lat.get('p50_ms', 0):.1f}ms "
            f"p70={lat.get('p70_ms', 0):.1f}ms "
            f"p100={lat.get('p100_ms', 0):.1f}ms",
            flush=True,
        )
        for row in (payload.get("slowest") or [])[:3]:
            print(
                f"  slow: {row.get('ms', 0):.1f}ms retrieve={row.get('retrieve_ms', 0):.1f}ms "
                f"{row.get('query', '')[:80]}",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001
        _bench_status = "error"
        _bench_error = str(exc)
        print(f"latency sweep failed: {exc}", flush=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _bench_status, _bench_error
    idx = settings.index_dir
    if (idx / "encoder.meta.json").exists() or (idx / "READY").exists():
        try:
            rag.load(idx)
        except Exception as exc:  # noqa: BLE001
            _bench_status = "error"
            _bench_error = f"index load failed: {exc}"
            print(_bench_error, flush=True)
        else:
            threading.Thread(
                target=_run_startup_sweep, daemon=True, name="latency-sweep"
            ).start()
    else:
        _bench_status = "error"
        _bench_error = "index not found"
    yield


app = FastAPI(title="Echo", version="0.1.0", lifespan=lifespan, default_response_class=ORJSONResponse)
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
    slm = rag.slm.ready if rag.harness is not None else False
    return {
        "ok": True,
        "ready": ready,
        "sarvam": bool(settings.sarvam_key_list()),
        "sarvam_keys": len(settings.sarvam_key_list()),
        "slm": slm,
        "stats": stats,
        "sla_ms": settings.sla_ms,
    }


@app.post("/api/ask")
def ask(body: AskBody) -> dict[str, Any]:
    if rag.harness is None:
        raise HTTPException(503, "Index not built. Run: python scripts/ingest.py")
    return _serialize(_attach_stt(_ask_sync(body.query), body.stt_ms))


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
        tr = transcribe(
            audio,
            filename=file.filename or "audio.webm",
            content_type=file.content_type or "audio/webm",
        )
    except SarvamError as exc:
        raise HTTPException(400, str(exc)) from exc
    result = _attach_stt(_ask_sync(tr.text), tr.ms)
    result.transcript = tr.text
    if tr.language_code:
        result.detected_language = tr.language_code
    return _serialize(result)


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    return {
        "live": summarize(_live_latencies),
        "bench": _bench,
        "bench_status": _bench_status,
        "bench_error": _bench_error,
        "bench_n": max(10, int(settings.bench_n or 120)),
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
