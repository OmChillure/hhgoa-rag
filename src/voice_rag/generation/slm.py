"""Tiny local SLM that rewrites an extractive span.

Prefill is the query + one span (not the retrieved passages). Decode is
capped so this can fit in the leftover ~100 ms budget. If the model is
missing or over the deadline, the harness keeps the extractive span.
"""

from __future__ import annotations

import time

from voice_rag.config import settings


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
            self._model = self._load_model(name)
            self.ready = True
            print(f"slm: ready ({self._kind})", flush=True)
        except Exception as exc:  # noqa: BLE001
            self.ready = False
            self._tok = None
            self._model = None
            print(f"slm: unavailable ({exc})", flush=True)

    def _load_model(self, name: str):
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

    def rewrite(self, query: str, span: str, timeout_ms: float | None = None) -> str:
        if not self.ready or self._tok is None or self._model is None:
            return ""
        fact = " ".join((span or "").split())
        question = " ".join((query or "").split())
        if not fact or not question:
            return ""
        if len(fact) > 280:
            cut = fact[:280]
            fact = cut.rsplit(" ", 1)[0] or cut
        prompt = (
            "Answer the question in one short sentence. "
            "Use only the fact. Do not add information.\n"
            f"Question: {question}\n"
            f"Fact: {fact}"
        )
        budget = float(timeout_ms if timeout_ms is not None else settings.slm_timeout_ms)
        deadline = time.perf_counter() + max(8.0, budget) / 1000.0
        try:
            encoded = self._tok(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=192,
            )
            generate_kw: dict = {
                "max_new_tokens": max(4, int(settings.slm_max_new_tokens)),
                "num_beams": 1,
                "do_sample": False,
                "use_cache": True,
            }
            try:
                from transformers import StoppingCriteria, StoppingCriteriaList

                class _Deadline(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs) -> bool:
                        return time.perf_counter() >= deadline

                generate_kw["stopping_criteria"] = StoppingCriteriaList([_Deadline()])
            except Exception:  # noqa: BLE001
                pass

            if self._kind == "torch":
                import torch

                prev = torch.get_num_threads()
                torch.set_num_threads(1)
                try:
                    with torch.inference_mode():
                        out = self._model.generate(**encoded, **generate_kw)
                finally:
                    torch.set_num_threads(prev)
            else:
                out = self._model.generate(**encoded, **generate_kw)

            text = self._tok.decode(out[0], skip_special_tokens=True).strip()
        except Exception as exc:  # noqa: BLE001
            print(f"slm: generate failed ({exc})", flush=True)
            return ""
        if not text or text.casefold() == fact.casefold():
            return text if text else ""
        if len(text) > 400:
            return ""
        return text
