from voice_rag.stt.convert import looks_wav, needs_convert, to_wav


def _tiny_wav() -> bytes:
    # 44-byte header + 2 samples of silence
    return (
        b"RIFF$\x00\x00\x00WAVEfmt "
        b"\x10\x00\x00\x00\x01\x00\x01\x00"
        b"\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00"
        b"data\x04\x00\x00\x00\x00\x00\x00\x00"
    )


def test_wav_passthrough():
    wav = _tiny_wav()
    assert looks_wav(wav)
    assert not needs_convert("clip.wav", "audio/wav", wav)
    out, name, ctype = to_wav(wav, "clip.wav", "audio/wav")
    assert out == wav
    assert name.endswith(".wav")
    assert ctype == "audio/wav"


def test_webm_needs_convert():
    fake = b"\x1a\x45\xdf\xa3not-really-webm"
    assert needs_convert("clip.webm", "audio/webm;codecs=opus", fake)
    # without a real webm, ffmpeg may fail and we return original
    out, _, _ = to_wav(fake, "clip.webm", "audio/webm;codecs=opus")
    assert isinstance(out, bytes)
