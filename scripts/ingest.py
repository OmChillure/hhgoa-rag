#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_rag.config import settings  # noqa: E402
from voice_rag.ingest.build import ingest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Vaani multi-strategy index.")
    parser.add_argument("--parquet", type=Path, default=settings.parquet_path)
    parser.add_argument("--out", type=Path, default=settings.index_dir)
    parser.add_argument("--n", type=int, default=settings.ingest_examples)
    args = parser.parse_args()
    meta = ingest(parquet=args.parquet, out_dir=args.out, n_examples=args.n)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
