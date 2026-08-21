"""Structured harness around retrieval + answering.

Not a raw prompt-in / text-out call. Every turn is a tool graph with:
- typed inputs/outputs
- explicit retries and fallbacks
- a recorded trace
- first-class refusal

Fast path: compiled DAG, extractive reader, then a tiny SLM rewrite.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Protocol

from voice_rag.config import settings
from voice_rag.generation.extractive import extract
from voice_rag.generation.slm import SpanRewriter, rewrite_is_faithful
from voice_rag.guardrails.policy import (
    check_grounding,
    check_input,
    check_retrieval,
    refusal_text,
)
from voice_rag.latency.metrics import Stopwatch
from voice_rag.textutil import detect_language, infer_query_type
from voice_rag.types import Answer, GuardDecision, Hit, PipelineResult


class _Searcher(Protocol):
    def search(self, query: str, language: str = "en", query_type: str = "UNKNOWN") -> list[Hit]: ...


Mode = Literal["fast"]


class ToolError(Exception):
    def __init__(self, tool: str, message: str) -> None:
        super().__init__(message)
        self.tool = tool
        self.message = message


class Harness:
    def __init__(self, retriever: _Searcher, slm: SpanRewriter | None = None) -> None:
        self.retriever = retriever
        self.slm = slm
        self.tools: dict[str, Callable[..., Any]] = {
            "safety_check": self.tool_safety_check,
            "classify_query": self.tool_classify,
            "retrieve": self.tool_retrieve,
            "extract_answer": self.tool_extract,
            "generate_answer": self.tool_generate,
            "ground_check": self.tool_ground,
            "refuse": self.tool_refuse,
        }

    def run(self, query: str, mode: Mode = "fast") -> PipelineResult:
        clock = Stopwatch()
        trace: list[dict[str, Any]] = []
        guards: list[GuardDecision] = []

        def call(name: str, **kwargs: Any) -> Any:
            retries = 1 if name in {"retrieve", "extract_answer"} else 0
            last_exc: Exception | None = None
            for attempt in range(retries + 1):
                with clock.span(name if attempt == 0 else f"{name}:retry{attempt}"):
                    try:
                        result = self.tools[name](**kwargs)
                        trace.append(
                            {
                                "tool": name,
                                "attempt": attempt,
                                "ok": True,
                                "input_keys": sorted(kwargs.keys()),
                            }
                        )
                        return result
                    except ToolError as exc:
                        last_exc = exc
                        trace.append(
                            {
                                "tool": name,
                                "attempt": attempt,
                                "ok": False,
                                "error": exc.message,
                            }
                        )
            raise last_exc or ToolError(name, "unknown")

        safety: GuardDecision = call("safety_check", query=query)
        guards.append(safety)
        if not safety.allowed:
            ans = call("refuse", reason=safety.reason)
            return self._finish(query, ans, guards, [], clock, trace, safety)

        meta = call("classify_query", query=query)
        hits: list[Hit] = call(
            "retrieve",
            query=query,
            language=meta["language"],
            query_type=meta["query_type"],
        )
        ret = check_retrieval(hits, query)
        guards.append(ret)
        if not ret.allowed:
            # Retry English only when the language shard returned nothing.
            # Low-confidence Indic hits still won't match the English index.
            if meta["language"] != "en" and not hits:
                hits = call("retrieve", query=query, language="en", query_type=meta["query_type"])
                ret = check_retrieval(hits, query)
                guards.append(ret)
            if not ret.allowed:
                ans = call("refuse", reason=ret.reason)
                return self._finish(query, ans, guards, hits, clock, trace, meta)

        extracted = call(
            "extract_answer",
            query=query,
            hits=hits,
            query_type=meta["query_type"],
        )
        ground = call("ground_check", answer=extracted["text"], hits=hits)
        guards.append(ground)

        if not ground.allowed:
            # recovery: try the next extractive span if any
            if extracted.get("alts"):
                for alt in extracted["alts"][:2]:
                    g_alt = check_grounding(alt, hits)
                    guards.append(g_alt)
                    if g_alt.allowed:
                        extracted["text"] = alt
                        ground = g_alt
                        break
            if not ground.allowed:
                ans = call("refuse", reason=ground.reason)
                ans.citations = hits[:3]
                return self._finish(query, ans, guards, hits, clock, trace, meta)

        text = extracted["text"]
        coverage = float((ground.details or {}).get("coverage") or 0.0)
        generated = call("generate_answer", query=query, span=text)
        gen_text = str((generated or {}).get("text") or "").strip()
        if gen_text and rewrite_is_faithful(gen_text, text, query):
            gen_ground = call("ground_check", answer=gen_text, hits=hits)
            guards.append(gen_ground)
            if gen_ground.allowed:
                text = gen_text
                coverage = float((gen_ground.details or {}).get("coverage") or coverage)

        ans = Answer(
            text=text,
            mode="generate",
            confidence=float(extracted.get("confidence") or 0.0),
            spans=extracted.get("spans") or [],
            citations=hits[:4],
            grounded=True,
            coverage=coverage,
            refused=False,
        )
        return self._finish(query, ans, guards, hits, clock, trace, meta)

    def _finish(
        self,
        query: str,
        answer: Answer,
        guards: list[GuardDecision],
        hits: list[Hit],
        clock: Stopwatch,
        trace: list[dict[str, Any]],
        meta: GuardDecision | dict[str, Any],
    ) -> PipelineResult:
        lang = "en"
        qtype = "UNKNOWN"
        if isinstance(meta, dict):
            lang = meta.get("language", "en")
            qtype = meta.get("query_type", "UNKNOWN")
        total = clock.total_ms
        result = PipelineResult(
            query=query,
            detected_language=lang,
            query_type=qtype,
            answer=answer,
            guardrails=guards,
            hits=hits,
            timings=clock.to_list(),
            total_ms=total,
            sla_ok=total < float(settings.sla_ms),
            harness_trace=trace,
        )
        return result

    # ----- tools -----
    def tool_safety_check(self, query: str) -> GuardDecision:
        return check_input(query)

    def tool_classify(self, query: str) -> dict[str, str]:
        return {
            "language": detect_language(query),
            "query_type": infer_query_type(query),
        }

    def tool_retrieve(self, query: str, language: str, query_type: str) -> list[Hit]:
        try:
            return self.retriever.search(query, language=language, query_type=query_type)
        except Exception as exc:  # noqa: BLE001
            raise ToolError("retrieve", str(exc)) from exc

    def tool_extract(self, query: str, hits: list[Hit], query_type: str) -> dict[str, Any]:
        text, spans, conf = extract(query, hits, query_type)
        alts = [s.text for s in spans[1:]]
        return {"text": text, "spans": spans, "confidence": conf, "alts": alts, "mode": "extractive"}

    def tool_generate(self, query: str, span: str) -> dict[str, Any]:
        if self.slm is None or not self.slm.ready:
            return {"text": "", "mode": "generate"}
        try:
            text = self.slm.rewrite(query, span, timeout_ms=settings.slm_timeout_ms)
        except Exception:  # noqa: BLE001
            return {"text": "", "mode": "generate"}
        return {"text": (text or "").strip(), "mode": "generate"}

    def tool_ground(self, answer: str, hits: list[Hit]) -> GuardDecision:
        return check_grounding(answer, hits)

    def tool_refuse(self, reason: str) -> Answer:
        return Answer(
            text=refusal_text(reason),
            mode="refuse",
            confidence=1.0,
            refused=True,
            refusal_reason=reason,
            grounded=True,
        )
