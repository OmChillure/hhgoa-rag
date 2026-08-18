from __future__ import annotations

from pathlib import Path

from voice_rag.config import settings
from voice_rag.harness.orchestrator import Harness, Mode
from voice_rag.retrieval.hybrid import HybridRetriever
from voice_rag.stt.sarvam import transcribe
from voice_rag.types import PipelineResult, StageTiming


class VoiceRAG:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.harness: Harness | None = None

    def load(self, directory: Path | None = None) -> None:
        directory = directory or settings.index_dir
        self.retriever.load(directory)
        self.harness = Harness(self.retriever)

    def ask(self, query: str, mode: Mode = "fast", *, use_cache: bool = True) -> PipelineResult:
        if self.harness is None:
            raise RuntimeError("index not loaded — run scripts/ingest.py first")
        return self.harness.run(query, mode=mode, use_cache=use_cache)

    def ask_audio(
        self,
        audio: bytes,
        filename: str = "audio.webm",
        content_type: str = "audio/webm",
        mode: Mode = "fast",
    ) -> PipelineResult:
        tr = transcribe(audio, filename=filename, content_type=content_type)
        result = self.ask(tr.text, mode=mode)
        result.transcript = tr.text
        result.stt_ms = tr.ms
        if not any(t.name == "stt" for t in result.timings):
            result.timings = [StageTiming(name="stt", ms=tr.ms), *result.timings]
        if tr.language_code:
            result.detected_language = tr.language_code
        return result
