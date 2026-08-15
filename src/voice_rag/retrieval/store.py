from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from voice_rag.types import Chunk


class FaissIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        if vectors.size == 0:
            return
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.ids.extend(ids)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        q = query.astype(np.float32).reshape(1, -1).copy()
        faiss.normalize_L2(q)
        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(q, k)
        out: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], idxs[0], strict=False):
            if idx < 0:
                continue
            out.append((self.ids[int(idx)], float(score)))
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path.with_suffix(".faiss")))
        path.with_suffix(".ids.json").write_text(json.dumps(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, dim: int) -> "FaissIndex":
        obj = cls(dim)
        obj.index = faiss.read_index(str(path.with_suffix(".faiss")))
        obj.ids = json.loads(path.with_suffix(".ids.json").read_text(encoding="utf-8"))
        obj.dim = obj.index.d
        return obj


def dump_chunks(path: Path, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(c.model_dump_json() for c in chunks),
        encoding="utf-8",
    )


def load_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunks.append(Chunk.model_validate_json(line))
    return chunks
