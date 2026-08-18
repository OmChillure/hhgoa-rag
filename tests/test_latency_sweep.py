from pathlib import Path
from types import SimpleNamespace

from voice_rag.latency.bench import load_bench_queries, run_latency_sweep
from voice_rag.latency.metrics import summarize


def test_summarize_percentiles_are_not_the_min():
    values = [float(i) for i in range(1, 101)]
    stats = summarize(values)
    assert stats["n"] == 100
    assert stats["p50_ms"] == 50.5
    assert stats["p70_ms"] == 70.3
    assert stats["p100_ms"] == 100.0
    assert stats["min_ms"] == 1.0
    assert stats["under_200ms_pct"] == 100.0


def test_load_bench_queries_from_sqlite(tmp_path: Path):
    import sqlite3

    db = sqlite3.connect(tmp_path / "passages.db")
    db.execute("CREATE TABLE queries (query_id INTEGER, en_query TEXT, en_answer TEXT)")
    db.executemany(
        "INSERT INTO queries VALUES (?, ?, ?)",
        [
            (1, "what is quartz", "a mineral"),
            (2, "what is a corporation?", "a company"),
            (3, "what is quartz", "dup"),
            (4, "", "skip"),
            (5, "where is pillager mn", "minnesota"),
        ],
    )
    db.commit()
    db.close()

    qs = load_bench_queries(tmp_path, 10, seed=0)
    assert len(qs) == 3
    assert "what is quartz" in qs
    assert "" not in qs


def test_load_bench_queries_prefers_holdout_json(tmp_path: Path):
    (tmp_path / "holdout_queries.json").write_text(
        '[{"en_query": "alpha"}, {"query": "beta"}, {"en_query": "alpha"}]',
        encoding="utf-8",
    )
    qs = load_bench_queries(tmp_path, 5, seed=0)
    assert set(qs) == {"alpha", "beta"}


def test_sweep_skips_warmup_and_aggregates_stages():
    calls: list[str] = []

    def ask(q: str):
        calls.append(q)
        idx = len(calls)
        return SimpleNamespace(
            total_ms=10.0 * idx,
            timings=[
                SimpleNamespace(name="retrieve", ms=8.0 * idx),
                SimpleNamespace(name="extract_answer", ms=2.0),
            ],
        )

    queries = [f"q{i}" for i in range(12)]
    payload = run_latency_sweep(ask, queries, sla_ms=200.0, warmup=2, passes=1)

    # two warm-ups (q0, q1) then all 12 recorded
    assert calls[:2] == ["q0", "q1"]
    assert payload["n_queries"] == 12
    assert payload["source"] == "startup_sweep"
    assert payload["latency"]["n"] == 12
    # recorded totals are 10*(3..14) because warmup already incremented idx
    assert payload["latency"]["p100_ms"] == 140.0
    assert payload["latency"]["min_ms"] == 30.0
    assert "retrieve" in payload["stages"]
    assert payload["stages"]["retrieve"]["n"] == 12
    assert payload["latency"]["p50_ms"] != payload["latency"]["p100_ms"]
    assert payload["slowest"][0]["ms"] == 140.0


def test_sweep_prime_pass_is_not_recorded():
    calls: list[str] = []

    def ask(q: str):
        calls.append(q)
        return SimpleNamespace(total_ms=20.0, timings=[])

    queries = ["a", "b", "c"]
    payload = run_latency_sweep(ask, queries, sla_ms=200.0, warmup=0, passes=2)
    # one prime pass + one recorded pass
    assert calls == ["a", "b", "c", "a", "b", "c"]
    assert payload["n_queries"] == 3
    assert payload["latency"]["p100_ms"] == 20.0


def test_sweep_empty_when_every_ask_fails():
    def ask(_q: str):
        raise RuntimeError("boom")

    assert run_latency_sweep(ask, ["a", "b", "c"], sla_ms=200.0) == {}
