"""Sarvam Saaras v3 speech-to-text with API-key rotation.

If one key is out of credits, rate-limited, or rejected, the next key
in the pool is tried. A working key is sticky until it fails.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import httpx

from voice_rag.config import settings
from voice_rag.stt.convert import to_wav

log = logging.getLogger("echo.stt")

SARVAM_URL = "https://api.sarvam.ai/speech-to-text"
# Failures that mean "this key is bad / exhausted / throttled" — try the next.
ROTATE_STATUSES = frozenset({401, 402, 403, 429, 500, 502, 503, 504})


@dataclass
class Transcript:
    text: str
    language_code: str | None
    request_id: str | None
    ms: float
    key_index: int = 0


class SarvamError(RuntimeError):
    pass


class _KeyRing:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cursor = 0

    def keys(self) -> list[str]:
        return settings.sarvam_key_list()

    def ordered(self) -> list[tuple[int, str]]:
        keys = self.keys()
        if not keys:
            return []
        with self._lock:
            start = self._cursor % len(keys)
        return [(i, keys[i]) for i in range(start, len(keys))] + [
            (i, keys[i]) for i in range(start)
        ]

    def mark_good(self, index: int) -> None:
        with self._lock:
            self._cursor = index


_ring = _KeyRing()


def transcribe(audio: bytes, filename: str = "audio.webm", content_type: str = "audio/webm") -> Transcript:
    keys = _ring.keys()
    if not keys:
        raise SarvamError(
            "No Sarvam key set. Add SARVAM_API_KEY or SARVAM_API_KEY_2..5 to .env."
        )
    if not audio:
        raise SarvamError("empty audio")

    t0 = time.perf_counter()
    audio, filename, content_type = to_wav(audio, filename, content_type)
    last_err: Exception | None = None
    tried: list[int] = []

    for index, key in _ring.ordered():
        if index in tried:
            continue
        tried.append(index)
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    SARVAM_URL,
                    headers={"api-subscription-key": key},
                    files={"file": (filename, audio, content_type)},
                    data={
                        "model": settings.sarvam_model,
                        "mode": settings.sarvam_mode,
                        "language_code": "unknown",
                    },
                )
            if resp.status_code in ROTATE_STATUSES:
                last_err = SarvamError(
                    f"key {index + 1}/{len(keys)} → HTTP {resp.status_code}"
                )
                log.warning("sarvam rotate: %s", last_err)
                continue
            if resp.status_code >= 400:
                raise SarvamError(f"Sarvam {resp.status_code}: {resp.text[:300]}")
            payload = resp.json()
            text = (payload.get("transcript") or "").strip()
            if not text:
                raise SarvamError("Sarvam returned an empty transcript")
            _ring.mark_good(index)
            return Transcript(
                text=text,
                language_code=payload.get("language_code"),
                request_id=payload.get("request_id"),
                ms=(time.perf_counter() - t0) * 1000.0,
                key_index=index,
            )
        except httpx.HTTPError as exc:
            last_err = exc
            log.warning("sarvam key %s network error: %s", index + 1, exc)
            continue

    raise SarvamError(
        f"all {len(keys)} Sarvam key(s) failed"
        + (f": {last_err}" if last_err else "")
    )
