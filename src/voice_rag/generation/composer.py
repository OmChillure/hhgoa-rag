"""Gemini composer for the optional quality path."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from voice_rag.config import settings
from voice_rag.types import Hit

log = logging.getLogger("vaani.composer")

SYSTEM = """You are Vaani, a grounded question-answering system over the MSMARCO-XI corpus.
Rules:
- Answer ONLY from the provided passages.
- If the passages do not contain the answer, say you do not know.
- Keep answers short (1-3 sentences).
- Do not invent numbers, names, or dates that are not in the passages.
- Return JSON: {"answer": str, "grounded": bool, "citation_parent_ids": [str]}
"""

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def active_provider() -> str | None:
    return "gemini" if settings.gemini_api_key else None


def _payload(query: str, hits: list[Hit]) -> str:
    ctx = [
        {"parent_id": h.chunk.parent_id, "text": h.parent_text or h.chunk.text}
        for h in hits[:5]
    ]
    return json.dumps({"query": query, "passages": ctx}, ensure_ascii=False)


def _parse_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"answer": text, "grounded": False, "citation_parent_ids": []}
    if isinstance(data, dict) and data.get("answer"):
        data["_provider"] = None
        return data
    return None


def compose(query: str, hits: list[Hit]) -> dict[str, Any] | None:
    if not settings.gemini_api_key:
        return None
    out = _gemini(query, hits)
    if out:
        out["_provider"] = "gemini"
    return out


def _gemini(query: str, hits: list[Hit]) -> dict[str, Any] | None:
    url = GEMINI_URL.format(model=settings.gemini_model)
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": _payload(query, hits)}]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }
    try:
        with httpx.Client(timeout=settings.gemini_timeout_s) as client:
            resp = client.post(url, params={"key": settings.gemini_api_key}, json=body)
        if resp.status_code >= 400:
            log.warning("gemini %s: %s", resp.status_code, resp.text[:300])
            return None
        payload = resp.json()
        text = (
            payload.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        return _parse_json(text)
    except Exception as exc:  # noqa: BLE001
        log.warning("gemini compose failed: %s", exc)
        return None
