"""Build one bm25s index per language for the sqlite-backed vaani_index.

A single BM25 index over all 7.77M passages scores the *entire* corpus as
a dense array on every query (that's how bm25s's numpy backend works,
regardless of query rarity) — about 900ms/query at this corpus size, far
over the 200ms SLA. Sharding by language cuts the per-query array to
~550K docs (1/14th), which should land comfortably under budget.

hybrid.py routes each query to its detected-language shard (falling back
to "en" when a shard doesn't exist or the query has no in-shard hits).

Saved via bm25s's native save() format (not a raw pickle) so each shard
can be loaded with mmap=True at query time.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import bm25s
from bm25s.tokenization import Tokenizer

from voice_rag.textutil import WORD_RE

INDEX_DIR = Path(__file__).resolve().parent.parent / "vaani_index"
DB_PATH = INDEX_DIR / "passages.db"
OUT_DIR = INDEX_DIR / "bm25"
BATCH = 100_000


def _splitter(text: str) -> list[str]:
    return WORD_RE.findall(text or "")


def build_shard(con: sqlite3.Connection, language: str, total: int) -> None:
    out_dir = OUT_DIR / language
    if out_dir.exists():
        print(f"[{language}] already built, skipping", flush=True)
        return

    t0 = time.perf_counter()
    tokenizer = Tokenizer(lower=True, splitter=_splitter, stopwords=None, stemmer=None)
    ids: list[str] = []

    def row_iter():
        cur = con.execute(
            "SELECT pid, text FROM passages WHERE language = ? ORDER BY rowid", (language,)
        )
        n = 0
        while True:
            rows = cur.fetchmany(BATCH)
            if not rows:
                break
            for row in rows:
                ids.append(row["pid"])
                yield row["text"]
            n += len(rows)
            print(f"[{language}] streamed {n}/{total} ({time.perf_counter()-t0:.1f}s)", flush=True)

    tokenized = tokenizer.tokenize(
        row_iter(), update_vocab=True, return_as="tuple", length=total, show_progress=False
    )

    print(f"[{language}] tokenized {total} in {time.perf_counter()-t0:.1f}s, indexing...", flush=True)
    bm25 = bm25s.BM25()
    bm25.index(tokenized, show_progress=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    bm25.save(str(out_dir), show_progress=False)
    (out_dir / "doc_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    print(f"[{language}] done in {time.perf_counter()-t0:.1f}s, {total} docs", flush=True)


def main() -> None:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    rows = con.execute(
        "SELECT language, COUNT(*) AS n FROM passages GROUP BY language ORDER BY n DESC"
    ).fetchall()
    languages = [(r["language"], r["n"]) for r in rows]
    print("languages:", languages, flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for lang, n in languages:
        build_shard(con, lang, n)
    print(f"all shards done in {time.perf_counter()-t0:.1f}s total", flush=True)


if __name__ == "__main__":
    main()
