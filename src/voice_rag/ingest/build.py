from __future__ import annotations

import json
from pathlib import Path

from voice_rag.chunking.ensemble import EnsembleChunker
from voice_rag.config import settings
from voice_rag.ingest.dataset import build_corpus, dump_json, load_rows, split_queries
from voice_rag.retrieval.hybrid import HybridRetriever


def ingest(
    parquet: Path | None = None,
    out_dir: Path | None = None,
    n_examples: int | None = None,
) -> dict:
    parquet = parquet or settings.parquet_path
    out_dir = out_dir or settings.index_dir
    n_examples = n_examples or settings.ingest_examples

    print(f"loading {n_examples} MSMARCO-XI rows from {parquet}", flush=True)
    rows = load_rows(parquet, limit=n_examples)
    n_rows = len(rows)
    print(f"rows={n_rows} — building corpus", flush=True)
    corpus = build_corpus(rows, languages=settings.languages)
    print(f"corpus {corpus['stats']}", flush=True)
    train_q, holdout_q = split_queries(corpus["queries"], settings.holdout_queries)
    del rows

    print("chunking (multi-strategy)…", flush=True)
    chunker = EnsembleChunker()
    chunks, parents = chunker.chunk_many(corpus["parents"])
    print(
        f"chunks={len(chunks)} parents={len(parents)} {EnsembleChunker.strategy_counts(chunks)}",
        flush=True,
    )

    print("indexing BM25 + LSA/FAISS…", flush=True)
    retriever = HybridRetriever()
    retriever.build(chunks, parents)
    print("writing index…", flush=True)
    retriever.save(out_dir)

    dump_json(out_dir / "holdout_queries.json", holdout_q)
    dump_json(out_dir / "ingest_meta.json", {
        "parquet": str(parquet),
        "examples": n_rows,
        "corpus": corpus["stats"],
        "chunks": len(chunks),
        "strategies": EnsembleChunker.strategy_counts(chunks),
        "holdout": len(holdout_q),
        "retriever": retriever.stats(),
    })
    (out_dir / "READY").write_text("ok\n", encoding="utf-8")
    return json.loads((out_dir / "ingest_meta.json").read_text(encoding="utf-8"))
