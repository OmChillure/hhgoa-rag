from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np

from voice_rag.types import StageTiming


class Stopwatch:
    def __init__(self) -> None:
        self.stages: list[StageTiming] = []

    @contextmanager
    def span(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            self.stages.append(StageTiming(name=name, ms=ms))

    @property
    def total_ms(self) -> float:
        return sum(s.ms for s in self.stages)

    def to_list(self) -> list[StageTiming]:
        return list(self.stages)


def percentile(values: Iterable[float], p: float) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, p))


def summarize(latencies_ms: list[float], sla_ms: float = 170.0) -> dict:
    under = sum(1 for x in latencies_ms if x < sla_ms)
    n = len(latencies_ms)
    return {
        "n": n,
        "mean_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "p50_ms": percentile(latencies_ms, 50),
        "p70_ms": percentile(latencies_ms, 70),
        "p90_ms": percentile(latencies_ms, 90),
        "p100_ms": percentile(latencies_ms, 100),
        "min_ms": float(min(latencies_ms)) if latencies_ms else 0.0,
        "max_ms": float(max(latencies_ms)) if latencies_ms else 0.0,
        "under_200ms": under,
        "under_200ms_pct": (100.0 * under / n) if n else 0.0,
        "sla_ms": sla_ms,
    }


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
