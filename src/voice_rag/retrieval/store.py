from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class FaissIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []

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
            if self.ids:
                out.append((self.ids[int(idx)], float(score)))
            else:
                # IndexIDMap: idx is the sqlite rowid we stored at build.
                out.append((str(int(idx)), float(score)))
        return out

    @classmethod
    def load(cls, path: Path, dim: int, nprobe: int | None = None) -> "FaissIndex":
        obj = cls(dim)
        obj.index = faiss.read_index(str(path.with_suffix(".faiss")))
        ids_json = path.with_suffix(".ids.json")
        ids_npy = path.with_suffix(".ids.npy")
        if ids_json.exists():
            obj.ids = json.loads(ids_json.read_text(encoding="utf-8"))
        elif ids_npy.exists():
            obj.ids = np.load(ids_npy, allow_pickle=True).tolist()
        else:
            obj.ids = []
        obj.dim = obj.index.d
        probe = 8 if nprobe is None else int(nprobe)
        try:
            faiss.extract_index_ivf(obj.index).nprobe = probe
        except RuntimeError:
            if hasattr(obj.index, "nprobe"):
                obj.index.nprobe = probe
        from voice_rag.config import settings

        threads = max(1, int(getattr(settings, "faiss_threads", 1) or 1))
        faiss.omp_set_num_threads(threads)
        return obj
