"""Vaani benchmark suite.

  Benchmark 1  Pure vector retrieval (FAISS search only)
  Benchmark 2  Full end-to-end RAG (retrieval + answer generation)
  Benchmark 3  Multilingual latency breakdown

Usage:
    PYTHONPATH=src python benchmark.py                 # all three, in-process
    PYTHONPATH=src python benchmark.py --http http://127.0.0.1:8080
    PYTHONPATH=src python benchmark.py -n 200 --json data/reports/benchmark.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from voice_rag.config import settings
from voice_rag.textutil import content_tokens

NO_ANSWER = "No Answer Present."

# Benchmark 3 probes. One natural question per script, plus the English control.
MULTILINGUAL_PROBES = [
    ("English", "en", "what is the capital of india"),
    ("Hindi (हिन्दी)", "hi", "भारत की राजधानी क्या है"),
    ("Marathi (मराठी)", "mr", "भारताची राजधानी कोणती आहे"),
    ("Bengali (বাংলা)", "bn", "ভারতের রাজধানী কোথায়"),
    ("Gujarati (ગુજરાતી)", "gu", "ભારતની રાજધાની કઈ છે"),
    ("Tamil (தமிழ்)", "ta", "இந்தியாவின் தலைநகரம் எது"),
    ("Kannada (ಕನ್ನಡ)", "kn", "ಭಾರತದ ರಾಜಧಾನಿ ಯಾವುದು"),
    ("Malayalam (മലയാളം)", "ml", "ഇന്ത്യയുടെ തലസ്ഥാനം ഏതാണ്"),
    ("Punjabi (ਪੰਜਾਬੀ)", "pa", "ਭਾਰਤ ਦੀ ਰਾਜਧਾਨੀ ਕੀ ਹੈ"),
    ("Odia (ଓଡ଼ିଆ)", "or", "ଭାରତର ରାଜଧାନୀ କଣ"),
    ("Urdu (اردو)", "ur", "بھارت کا دارالحکومت کیا ہے"),
    ("Assamese (অসমীয়া)", "as", "ভাৰতৰ ৰাজধানী কি"),
    ("Nepali (नेपाली)", "ne", "भारतको राजधानी के हो"),
    ("Sanskrit (संस्कृतम्)", "sa", "भारतस्य राजधानी का अस्ति"),
]


# ---------------------------------------------------------------- plumbing


class HttpRAG:
    """Drives the already-running API so we don't reload the 12GB index."""

    def __init__(self, base_url: str) -> None:
        import httpx

        self.base = base_url.rstrip("/")
        self.client = httpx.Client(timeout=120.0)
        health = self.client.get(f"{self.base}/api/health").json()
        if not health.get("ready"):
            raise RuntimeError(f"server at {self.base} has no index loaded")
        self._stats = health.get("stats", {})
        self.retriever = None  # benchmark 1 needs in-process access

    def stats(self) -> dict:
        return self._stats

    def ask(self, query: str):
        r = self.client.post(f"{self.base}/api/ask", json={"query": query})
        r.raise_for_status()
        return _as_result(r.json())


def _as_result(payload: dict):
    """Rebuild the attribute access the report code expects from JSON."""
    ans = payload.get("answer") or {}
    hits = [
        SimpleNamespace(
            parent_text=h.get("parent_text") or "",
            chunk=SimpleNamespace(text=(h.get("chunk") or {}).get("text", "")),
            origin=h.get("origin", ""),
        )
        for h in payload.get("hits") or []
    ]
    return SimpleNamespace(
        answer=SimpleNamespace(
            text=ans.get("text", ""),
            refused=bool(ans.get("refused")),
            refusal_reason=ans.get("refusal_reason", ""),
            confidence=float(ans.get("confidence") or 0.0),
            coverage=float(ans.get("coverage") or 0.0),
        ),
        hits=hits,
        timings=[
            SimpleNamespace(name=t.get("name", "?"), ms=float(t.get("ms") or 0.0))
            for t in payload.get("timings") or []
        ],
        total_ms=float(payload.get("total_ms") or 0.0),
        detected_language=payload.get("detected_language", "en"),
        query_type=payload.get("query_type", "UNKNOWN"),
    )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


def load_gold(index_dir: Path) -> list[dict]:
    db_path = index_dir / "passages.db"
    if not db_path.exists():
        return []
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT query_id, en_query, en_answer FROM queries "
        "WHERE en_query IS NOT NULL AND TRIM(en_query) != ''"
    ).fetchall()
    db.close()
    return [
        {
            "query_id": r["query_id"],
            "query": r["en_query"].strip(),
            "gold": (r["en_answer"] or "").strip(),
            "answerable": (r["en_answer"] or "").strip() != NO_ANSWER,
        }
        for r in rows
    ]


def token_f1(pred: str, gold: str) -> float:
    """SQuAD-style token F1 between predicted span and gold answer."""
    p, g = content_tokens(pred), content_tokens(gold)
    if not p or not g:
        return 0.0
    counts: dict[str, int] = defaultdict(int)
    for t in p:
        counts[t] += 1
    common = 0
    for t in g:
        if counts[t] > 0:
            counts[t] -= 1
            common += 1
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def gold_recall(gold: str, result) -> float:
    """Fraction of gold answer tokens present in any retrieved passage."""
    g = set(content_tokens(gold))
    if not g:
        return 0.0
    pooled: set[str] = set()
    for hit in result.hits:
        pooled |= set(content_tokens(hit.parent_text or hit.chunk.text))
    return len(g & pooled) / len(g)


def row(label: str, value: float, unit: str = "ms", width: int = 26) -> str:
    return f"    - {label:<{width}}{value:>8.2f} {unit}"


# ------------------------------------------------------- benchmark 1


def bench_pure_retrieval(rag, queries: list[str], n: int) -> dict:
    """FAISS vector search in isolation: encode excluded, search only."""
    print("\n" + "=" * 72)
    print("  Benchmark 1: Pure Vector Retrieval (FAISS search only)")
    print("=" * 72)

    retriever = getattr(rag, "retriever", None)
    if retriever is None or retriever.lsa_index is None or retriever.st_model is None:
        print("    skipped -- needs in-process index (drop --http to run this)")
        return {}

    # Pre-encode so we time the index, not the sentence-transformer.
    vectors = [
        retriever.st_model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
        for q in queries[: min(n, len(queries))]
    ]
    for v in vectors[:5]:  # warm the index
        retriever.lsa_index.search(v, settings.dense_top_k)

    search_ms: list[float] = []
    for i in range(n):
        vec = vectors[i % len(vectors)]
        t0 = time.perf_counter()
        retriever.lsa_index.search(vec, settings.dense_top_k)
        search_ms.append((time.perf_counter() - t0) * 1000.0)

    out = {
        "n": n,
        "avg_ms": statistics.mean(search_ms),
        "p50_ms": percentile(search_ms, 50),
        "p95_ms": percentile(search_ms, 95),
        "p99_ms": percentile(search_ms, 99),
    }
    print(row("AVG Latency:", out["avg_ms"]))
    print(row("P50 (Median):", out["p50_ms"]))
    print(row("P95:", out["p95_ms"]))
    print(row("P99:", out["p99_ms"]))
    return out


# ------------------------------------------------------- benchmark 2


def bench_end_to_end(rag, gold: list[dict], n: int) -> tuple[dict, list[dict]]:
    """Full harness: safety -> classify -> retrieve -> extract -> ground."""
    print("\n" + "=" * 72)
    print("  Benchmark 2: Full End-to-End RAG (Retrieval + Answer Generation)")
    print("=" * 72)

    rag.ask(gold[0]["query"])  # warmup

    total_ms: list[float] = []
    stage_ms: dict[str, list[float]] = defaultdict(list)
    ttft: list[float] = []
    records: list[dict] = []
    cache_probe: list[float] = []

    for i in range(n):
        item = gold[i % len(gold)]
        t0 = time.perf_counter()
        res = rag.ask(item["query"])
        wall = (time.perf_counter() - t0) * 1000.0

        total_ms.append(wall)
        stages = {t.name: t.ms for t in res.timings}
        for name, ms in stages.items():
            stage_ms[name].append(ms)
        # This pipeline is extractive, not streaming: the answer span exists
        # the moment extract_answer returns, so "time to first token" is
        # everything up to and including extraction.
        ttft.append(sum(
            ms for name, ms in stages.items()
            if name in {"safety_check", "classify_query", "retrieve", "extract_answer"}
        ))

        answerable = item["answerable"] and bool(item["gold"])
        records.append({
            "query_id": item["query_id"],
            "query": item["query"],
            "gold": item["gold"],
            "answerable": item["answerable"],
            "answer": res.answer.text,
            "refused": res.answer.refused,
            "refusal_reason": res.answer.refusal_reason,
            "confidence": res.answer.confidence,
            "coverage": res.answer.coverage,
            "language": res.detected_language,
            "query_type": res.query_type,
            "n_hits": len(res.hits),
            "wall_ms": wall,
            "harness_ms": res.total_ms,
            "sla_ok": wall < settings.sla_ms,
            "token_f1": token_f1(res.answer.text, item["gold"]) if answerable else 0.0,
            "gold_recall": gold_recall(item["gold"], res) if answerable else 0.0,
        })
        if (i + 1) % 50 == 0:
            print(f"      {i + 1}/{n} queries...", flush=True)

    # Cache turnaround: repeat a query already served, measure the delta.
    repeat = gold[0]["query"]
    rag.ask(repeat)
    for _ in range(10):
        t0 = time.perf_counter()
        rag.ask(repeat)
        cache_probe.append((time.perf_counter() - t0) * 1000.0)

    out = {
        "n": n,
        "p50_ms": percentile(total_ms, 50),
        "p70_ms": percentile(total_ms, 70),
        "p90_ms": percentile(total_ms, 90),
        "p95_ms": percentile(total_ms, 95),
        "p99_ms": percentile(total_ms, 99),
        "p100_ms": max(total_ms),
        "mean_ms": statistics.mean(total_ms),
        "ttft_ms": percentile(ttft, 50),
        "cache_turnaround_ms": statistics.mean(cache_probe),
        "within_sla_pct": 100.0 * sum(1 for x in total_ms if x < settings.sla_ms) / n,
        "stages": {k: {"mean_ms": statistics.mean(v), "p95_ms": percentile(v, 95)}
                   for k, v in stage_ms.items()},
    }
    print(row("P50 Latency (Median):", out["p50_ms"]))
    print(row("P70 Latency:", out["p70_ms"]))
    print(row("P90 Latency:", out["p90_ms"]))
    print(row("P95 Latency:", out["p95_ms"]))
    print(row("P100 Latency (Max):", out["p100_ms"]))
    print(row("Mean (Average):", out["mean_ms"]))
    print(row("TTFT (Time to Token):", out["ttft_ms"]))
    print(row("Cache Turnaround:", out["cache_turnaround_ms"]))

    print("\n    Stage breakdown (mean ms):")
    for name, s in sorted(out["stages"].items(), key=lambda kv: -kv[1]["mean_ms"]):
        print(f"      {name:<20}{s['mean_ms']:>8.2f}  (p95 {s['p95_ms']:>7.2f})")
    return out, records


# ------------------------------------------------------- benchmark 3


def bench_multilingual(rag, repeats: int) -> dict:
    """Per-language latency, split into retrieval vs generation."""
    print("\n" + "=" * 72)
    print(f"  Benchmark 3: Multilingual Latency Breakdown "
          f"(Across {len(MULTILINGUAL_PROBES)} Languages)")
    print("=" * 72)

    results: dict[str, dict] = {}
    width = max(len(name) for name, _, _ in MULTILINGUAL_PROBES) + 2

    for name, want, query in MULTILINGUAL_PROBES:
        rag.ask(query)  # warm this shard
        totals, retr, gen, got, refused = [], [], [], "", 0
        for _ in range(repeats):
            t0 = time.perf_counter()
            res = rag.ask(query)
            totals.append((time.perf_counter() - t0) * 1000.0)
            stages = {t.name: t.ms for t in res.timings}
            retr.append(stages.get("retrieve", 0.0))
            gen.append(sum(v for k, v in stages.items() if k != "retrieve"))
            got = res.detected_language
            refused += int(res.answer.refused)
            answer = res.answer.text

        results[name] = {
            "code": want,
            "detected": got,
            "routed_ok": got == want,
            "total_ms": statistics.mean(totals),
            "retrieval_ms": statistics.mean(retr),
            "generation_ms": statistics.mean(gen),
            "refused": refused,
            "answer": answer,
        }
        r = results[name]
        flag = " " if r["routed_ok"] else "*"
        print(f"    - {name + ':':<{width}}{r['total_ms']:>8.2f} ms  "
              f"(Retrieval: {r['retrieval_ms']:>6.2f} ms | "
              f"Gen: {r['generation_ms']:>6.2f} ms){flag}")

    misrouted = [n for n, r in results.items() if not r["routed_ok"]]
    if misrouted:
        print(f"\n    * language misrouted by detect_language: {', '.join(misrouted)}")
    return results


# ------------------------------------------------------- quality


def report_quality(records: list[dict]) -> dict:
    print("\n" + "=" * 72)
    print("  Answer Quality (vs gold answers in the index)")
    print("=" * 72)

    answerable = [r for r in records if r["answerable"] and r["gold"]]
    unanswerable = [r for r in records if not r["answerable"]]
    out: dict = {}

    if answerable:
        f1s = [r["token_f1"] for r in answerable]
        recalls = [r["gold_recall"] for r in answerable]
        answered = [r for r in answerable if not r["refused"]]
        out = {
            "answerable": len(answerable),
            "answered_pct": 100.0 * len(answered) / len(answerable),
            "mean_token_f1": statistics.mean(f1s),
            "median_token_f1": percentile(f1s, 50),
            "f1_ge_50_pct": 100.0 * sum(1 for x in f1s if x >= 0.5) / len(f1s),
            "f1_ge_30_pct": 100.0 * sum(1 for x in f1s if x >= 0.3) / len(f1s),
            "gold_recall_at_k": statistics.mean(recalls),
        }
        print(f"    - {'Answerable queries:':<28}{len(answerable):>8}")
        print(f"    - {'Answered (not refused):':<28}{len(answered):>8}"
              f"  ({out['answered_pct']:.1f}%)")
        print(f"    - {'Mean token F1:':<28}{out['mean_token_f1']:>8.3f}")
        print(f"    - {'Median token F1:':<28}{out['median_token_f1']:>8.3f}")
        print(f"    - {'F1 >= 0.5:':<28}{out['f1_ge_50_pct']:>8.1f} %")
        print(f"    - {'F1 >= 0.3:':<28}{out['f1_ge_30_pct']:>8.1f} %")
        print(f"    - {'Gold recall@k (retrieval):':<28}{out['gold_recall_at_k']:>8.3f}")

    if unanswerable:
        refused = sum(1 for r in unanswerable if r["refused"])
        out["unanswerable"] = len(unanswerable)
        out["correct_refusal_pct"] = 100.0 * refused / len(unanswerable)
        print(f"\n    - {'Unanswerable queries:':<28}{len(unanswerable):>8}")
        print(f"    - {'Correctly refused:':<28}{refused:>8}"
              f"  ({out['correct_refusal_pct']:.1f}%)")
    return out


# ------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n", type=int, default=400, help="queries for benchmarks 1 & 2")
    ap.add_argument("--repeats", type=int, default=5, help="repeats per language in benchmark 3")
    ap.add_argument("--http", default=None, help="drive a running server instead of loading the index")
    ap.add_argument("--json", dest="json_out", default=None, help="write report JSON here")
    ap.add_argument("--only", default=None, help="comma-separated subset: 1,2,3")
    args = ap.parse_args()

    only = {x.strip() for x in args.only.split(",")} if args.only else {"1", "2", "3"}

    t0 = time.perf_counter()
    if args.http:
        print(f"Using running server at {args.http} ...", flush=True)
        rag = HttpRAG(args.http)
        stats = rag.stats()
    else:
        print("Loading index (sqlite + FAISS + BM25 shards)...", flush=True)
        from voice_rag.pipeline import VoiceRAG

        rag = VoiceRAG()
        rag.load()
        stats = rag.retriever.stats()
    load_s = time.perf_counter() - t0
    print(f"ready in {load_s:.1f}s | passages={stats.get('chunks'):,} "
          f"dim={stats.get('lsa_dim')} encoder={stats.get('encoder')}")

    gold = load_gold(settings.index_dir)
    if not gold:
        print("no gold queries found in index")
        sys.exit(1)
    print(f"gold queries: {len(gold)} "
          f"({sum(1 for g in gold if g['answerable'])} answerable)")

    n = min(args.n, len(gold))
    payload: dict = {"index": stats, "index_load_s": load_s, "sla_ms": settings.sla_ms}

    if "1" in only:
        payload["benchmark_1_pure_retrieval"] = bench_pure_retrieval(
            rag, [g["query"] for g in gold], n
        )

    records: list[dict] = []
    if "2" in only:
        e2e, records = bench_end_to_end(rag, gold, n)
        payload["benchmark_2_end_to_end"] = e2e

    if "3" in only:
        payload["benchmark_3_multilingual"] = bench_multilingual(rag, args.repeats)

    if records:
        payload["quality"] = report_quality(records)
        payload["records"] = records

    out_path = Path(args.json_out) if args.json_out else settings.reports_dir / "benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport written: {out_path}")

    e2e = payload.get("benchmark_2_end_to_end")
    if e2e:
        p95 = e2e["p95_ms"]
        print()
        if p95 <= settings.sla_ms:
            print(f"PASS: p95 {p95:.1f}ms within {settings.sla_ms:.0f}ms SLA "
                  f"({e2e['within_sla_pct']:.1f}% of queries under budget)")
        else:
            print(f"FAIL: p95 {p95:.1f}ms over {settings.sla_ms:.0f}ms SLA")
            sys.exit(1)


if __name__ == "__main__":
    main()
