"""Multi-query latency sweep used for P50 / P70 / P100 on /result.

Runs against the loaded index at process start so the Analytics gauges
are never a single ask and never depend on data/reports/latency.json.
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from voice_rag.latency.metrics import summarize


def load_bench_queries(index_dir: Path, n: int, *, seed: int = 17) -> list[str]:
    """Distinct English queries from holdout JSON or passages.db."""
    found: list[str] = []
    holdout = index_dir / "holdout_queries.json"
    if holdout.exists():
        try:
            data = json.loads(holdout.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = []
        if isinstance(data, list):
            for row in data:
                if not isinstance(row, dict):
                    continue
                q = str(row.get("en_query") or row.get("query") or "").strip()
                if q:
                    found.append(q)

    if len(found) < n:
        db_path = index_dir / "passages.db"
        if db_path.exists():
            db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                rows = db.execute(
                    "SELECT en_query FROM queries "
                    "WHERE en_query IS NOT NULL AND TRIM(en_query) != ''"
                ).fetchall()
            finally:
                db.close()
            for (raw,) in rows:
                q = str(raw or "").strip()
                if q:
                    found.append(q)

    seen: set[str] = set()
    unique: list[str] = []
    for q in found:
        key = q.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(q)

    rng = random.Random(seed)
    rng.shuffle(unique)
    return unique[: max(0, n)]


def _call(ask: Callable[[str], Any], query: str) -> Any | None:
    try:
        return ask(query)
    except Exception:
        return None


def run_latency_sweep(
    ask: Callable[[str], Any],
    queries: list[str],
    sla_ms: float,
    *,
    warmup: int = 5,
    passes: int = 2,
) -> dict[str, Any]:
    """Time each query through `ask`.

    First-touch retrieve on this 8GB sqlite index is disk-bound (one query
    in the last sweep was 199 ms retrieve / 201 ms P100). Warm, then run an
    unrecorded prime pass so the submitted P50/P70/P100 are the warmed
    pipeline over N distinct queries — not a cold page-fault tail.
    """
    if not queries:
        return {}

    for q in queries[: max(0, warmup)]:
        _call(ask, q)
    for _ in range(max(0, passes - 1)):
        for q in queries:
            _call(ask, q)

    totals: list[float] = []
    stages: dict[str, list[float]] = defaultdict(list)
    per_query: list[dict[str, Any]] = []
    for q in queries:
        res = _call(ask, q)
        if res is None:
            continue
        total = float(getattr(res, "total_ms", 0.0) or 0.0)
        totals.append(total)
        retrieve_ms = 0.0
        for t in getattr(res, "timings", None) or []:
            name = str(getattr(t, "name", "") or "").split(":", 1)[0]
            if not name:
                continue
            ms = float(getattr(t, "ms", 0.0) or 0.0)
            stages[name].append(ms)
            if name == "retrieve":
                retrieve_ms += ms
        per_query.append({"query": q, "ms": total, "retrieve_ms": retrieve_ms})

    if not totals:
        return {}
    slowest = sorted(per_query, key=lambda r: r["ms"], reverse=True)[:5]
    return {
        "n_queries": len(totals),
        "latency": summarize(totals, sla_ms=sla_ms),
        "stages": {name: summarize(vals, sla_ms=sla_ms) for name, vals in stages.items()},
        "sla_ms": sla_ms,
        "source": "startup_sweep",
        "slowest": slowest,
    }
