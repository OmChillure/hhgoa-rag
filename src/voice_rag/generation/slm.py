"""Tiny local SLM that always rewrites the extractive span.

Uses CTranslate2 INT8 when available (typical T5-small CPU path:
~20–40 ms vs 80–200 ms in HuggingFace). Falls back to ONNX / torch.
Prefill is query + one span, greedy decode, short output.
"""

from __future__ import annotations

from pathlib import Path

from voice_rag.config import settings


def _prompt(question: str, fact: str) -> str:
    return f"question: {question}\nfact: {fact}\nshort answer:"


def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= n:
        return text
    cut = text[:n]
    return cut.rsplit(" ", 1)[0] or cut


class SpanRewriter:
    def __init__(self) -> None:
        self.ready = False
        self._tok = None
        self._model = None
        self._kind = ""

    def load(self) -> None:
        if not settings.slm_enabled:
            print("slm: disabled", flush=True)
            return
        name = settings.slm_model
        print(f"slm: loading {name}…", flush=True)
        try:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(name)
            self._model = self._load_ct2(name) or self._load_hf(name)
            self.ready = self._model is not None
            if self.ready:
                self._warmup()
                print(f"slm: ready ({self._kind})", flush=True)
            else:
                print("slm: unavailable (no backend)", flush=True)
        except Exception as exc:  # noqa: BLE001
            self.ready = False
            self._tok = None
            self._model = None
            print(f"slm: unavailable ({exc})", flush=True)

    def _ct2_dir(self, name: str) -> Path:
        safe = name.replace("/", "_")
        return Path(settings.slm_cache_dir) / f"{safe}-ct2-int8"

    def _load_ct2(self, name: str):
        try:
            import ctranslate2
        except ImportError:
            print("slm: ctranslate2 not installed", flush=True)
            return None
        out = self._ct2_dir(name)
        try:
            if not (out / "model.bin").exists():
                print(f"slm: converting {name} → INT8 CTranslate2…", flush=True)
                from ctranslate2.converters import TransformersConverter

                out.parent.mkdir(parents=True, exist_ok=True)
                TransformersConverter(name).convert(
                    str(out),
                    quantization="int8",
                    force=True,
                )
            translator = ctranslate2.Translator(
                str(out),
                device="cpu",
                compute_type="int8",
                inter_threads=1,
                intra_threads=max(1, int(settings.slm_threads)),
            )
            self._kind = "ct2-int8"
            return translator
        except Exception as exc:  # noqa: BLE001
            print(f"slm: ctranslate2 skipped ({exc})", flush=True)
            return None

    def _load_hf(self, name: str):
        try:
            from optimum.onnxruntime import ORTModelForSeq2SeqLM

            model = ORTModelForSeq2SeqLM.from_pretrained(name, export=True)
            self._kind = "onnx"
            return model
        except Exception as exc:  # noqa: BLE001
            print(f"slm: onnx export skipped ({exc}); using torch", flush=True)
        from transformers import AutoModelForSeq2SeqLM

        model = AutoModelForSeq2SeqLM.from_pretrained(name)
        model.eval()
        self._kind = "torch"
        return model

    def _warmup(self) -> None:
        try:
            self.rewrite("what is x", "X is a thing used in tests.")
        except Exception:  # noqa: BLE001
            pass

    def rewrite(self, query: str, span: str, timeout_ms: float | None = None) -> str:
        if not self.ready or self._tok is None or self._model is None:
            return ""
        fact = _clip(span, 180)
        question = _clip(query, 120)
        if not fact or not question:
            return ""
        prompt = _prompt(question, fact)
        max_new = max(4, int(settings.slm_max_new_tokens))
        try:
            if self._kind == "ct2-int8":
                text = self._generate_ct2(prompt, max_new)
            else:
                text = self._generate_hf(prompt, max_new)
        except Exception as exc:  # noqa: BLE001
            print(f"slm: generate failed ({exc})", flush=True)
            return ""
        text = (text or "").strip()
        if len(text) > 400:
            return ""
        return text

    def _generate_ct2(self, prompt: str, max_new: int) -> str:
        ids = self._tok.encode(prompt, truncation=True, max_length=96)
        tokens = self._tok.convert_ids_to_tokens(ids)
        results = self._model.translate_batch(
            [tokens],
            beam_size=1,
            max_decoding_length=max_new,
            min_decoding_length=1,
            return_scores=False,
        )
        out_tokens = results[0].hypotheses[0]
        out_ids = self._tok.convert_tokens_to_ids(out_tokens)
        return self._tok.decode(out_ids, skip_special_tokens=True)

    def _generate_hf(self, prompt: str, max_new: int) -> str:
        encoded = self._tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=96,
        )
        generate_kw = {
            "max_new_tokens": max_new,
            "min_new_tokens": 1,
            "num_beams": 1,
            "do_sample": False,
            "use_cache": True,
        }
        if self._kind == "torch":
            import torch

            prev = torch.get_num_threads()
            torch.set_num_threads(max(1, int(settings.slm_threads)))
            try:
                with torch.inference_mode():
                    out = self._model.generate(**encoded, **generate_kw)
            finally:
                torch.set_num_threads(prev)
        else:
            out = self._model.generate(**encoded, **generate_kw)
        return self._tok.decode(out[0], skip_special_tokens=True)
