"""Normalize browser recordings to 16 kHz mono WAV for Sarvam.

Sarvam rejects audio/webm;codecs=opus (what Chrome MediaRecorder emits).
Accepted: wav, mp3, aac, aiff, flac, ogg, and raw PCM.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger("vaani.stt")

ACCEPTED_SUFFIX = {".wav", ".mp3", ".mpeg", ".aac", ".aiff", ".aif", ".flac", ".ogg", ".opus"}
ACCEPTED_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/aac",
    "audio/x-aac",
    "audio/aiff",
    "audio/flac",
    "audio/ogg",
    "audio/opus",
}


def looks_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def needs_convert(filename: str, content_type: str, data: bytes) -> bool:
    if looks_wav(data):
        return False
    suffix = Path(filename or "").suffix.lower()
    ctype = (content_type or "").split(";")[0].strip().lower()
    if suffix in ACCEPTED_SUFFIX or ctype in ACCEPTED_TYPES:
        return False
    return True


def to_wav(data: bytes, filename: str = "", content_type: str = "") -> tuple[bytes, str, str]:
    """Return (bytes, filename, content_type). Identity if already WAV/accepted."""
    if not needs_convert(filename, content_type, data) and looks_wav(data):
        name = filename if (filename or "").lower().endswith(".wav") else "clip.wav"
        return data, name, "audio/wav"
    if not needs_convert(filename, content_type, data):
        return data, filename or "audio.bin", content_type or "application/octet-stream"

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        log.warning("ffmpeg missing; sending original %s to Sarvam", content_type)
        return data, filename or "audio.bin", content_type or "application/octet-stream"

    suffix = Path(filename or "clip.webm").suffix or ".webm"
    with tempfile.TemporaryDirectory(prefix="vaani-stt-") as tmp:
        src = Path(tmp) / f"in{suffix}"
        dst = Path(tmp) / "out.wav"
        src.write_bytes(data)
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            err = ""
            if isinstance(exc, subprocess.CalledProcessError):
                err = (exc.stderr or b"").decode("utf-8", "ignore")[:240]
            log.warning("ffmpeg convert failed: %s %s", exc, err)
            return data, filename or "audio.bin", content_type or "application/octet-stream"
        wav = dst.read_bytes()
    if not looks_wav(wav):
        return data, filename or "audio.bin", content_type or "application/octet-stream"
    return wav, "clip.wav", "audio/wav"
