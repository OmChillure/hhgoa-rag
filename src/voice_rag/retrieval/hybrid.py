from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import bm25s

from voice_rag.config import settings
from voice_rag.retrieval.embeddings import LSAEncoder
from voice_rag.retrieval.store import FaissIndex, dump_chunks, load_chunks
from voice_rag.textutil import detect_language, infer_query_type, tokenize
from voice_rag.types import Chunk, ChunkStrategy, Hit


def rrf(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranks in rank_lists:
        for i, cid in enumerate(ranks):
            scores[cid] += 1.0 / (k + i + 1)
    return dict(scores)


class HybridRetriever:
    """Multi-index hybrid search.

    BM25 + LSA/FAISS, fused with RRF, then parent expansion.
    """

    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}
        self.parents: dict[str, str] = {}
        self.lsa = LSAEncoder(
            n_features=settings.tfidf_features,
            n_components=settings.lsa_components,
        )
        self.lsa_index: FaissIndex | None = None
        self.bm25 = None
        self.bm25_ids: list[str] = []
        self.by_lang: dict[str, list[str]] = defaultdict(list)
        self.by_strategy: dict[str, list[str]] = defaultdict(list)

    # ----- ingest -----
    def build(self, chunks: list[Chunk], parents: dict[str, str]) -> None:
        self.chunks = {c.chunk_id: c for c in chunks}
        self.parents = parents
        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        for c in chunks:
            self.by_lang[c.language].append(c.chunk_id)
            self.by_strategy[c.strategy.value].append(c.chunk_id)

        tokens = [tokenize(t) for t in texts]
        self.bm25 = bm25s.BM25()
        self.bm25.index(tokens, show_progress=False)
        self.bm25_ids = ids

        # Dense indexes stay on the most useful families so ingest
        # and query encoding stay inside a demo-friendly budget.
        dense_strats = {
            ChunkStrategy.PASSAGE,
            ChunkStrategy.SEMANTIC,
            ChunkStrategy.SENTENCE_WINDOW,
            ChunkStrategy.HIERARCHICAL_CHILD,
        }
        dense = [c for c in chunks if c.strategy in dense_strats]
        dense_ids = [c.chunk_id for c in dense]
        dense_texts = [c.text for c in dense]
        self.lsa.fit(dense_texts)
        self.lsa_index = FaissIndex(self.lsa.dim)
        self.lsa_index.add(self.lsa.encode(dense_texts), dense_ids)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        dump_chunks(directory / "chunks.jsonl", list(self.chunks.values()))
        (directory / "parents.json").write_text(
            json.dumps(self.parents, ensure_ascii=False), encoding="utf-8"
        )
        assert self.lsa_index is not None
        self.lsa_index.save(directory / "lsa")
        with (directory / "lsa_encoder.pkl").open("wb") as fh:
            pickle.dump(self.lsa, fh)
        with (directory / "bm25.pkl").open("wb") as fh:
            pickle.dump({"bm25": self.bm25, "ids": self.bm25_ids}, fh)
        (directory / "stats.json").write_text(
            json.dumps(self.stats(), indent=2), encoding="utf-8"
        )

    def load(self, directory: Path) -> None:
        chunks = load_chunks(directory / "chunks.jsonl")
        self.chunks = {c.chunk_id: c for c in chunks}
        self.parents = json.loads((directory / "parents.json").read_text(encoding="utf-8"))
        self.by_lang = defaultdict(list)
        self.by_strategy = defaultdict(list)
        for c in chunks:
            self.by_lang[c.language].append(c.chunk_id)
            self.by_strategy[c.strategy.value].append(c.chunk_id)
        with (directory / "lsa_encoder.pkl").open("rb") as fh:
            self.lsa = pickle.load(fh)
        self.lsa_index = FaissIndex.load(directory / "lsa", self.lsa.dim)
        with (directory / "bm25.pkl").open("rb") as fh:
            payload = pickle.load(fh)
            self.bm25 = payload["bm25"]
            self.bm25_ids = payload["ids"]

    def stats(self) -> dict:
        return {
            "chunks": len(self.chunks),
            "parents": len(self.parents),
            "languages": {k: len(v) for k, v in self.by_lang.items()},
            "strategies": {k: len(v) for k, v in self.by_strategy.items()},
            "lsa_dim": self.lsa.dim,
        }

    # ----- query -----
    def search(
        self,
        query: str,
        *,
        language: str | None = None,
        query_type: str | None = None,
        top_k: int | None = None,
    ) -> list[Hit]:
        language = language or detect_language(query)
        query_type = query_type or infer_query_type(query)
        top_k = top_k or settings.fused_top_k

        lists: list[list[str]] = []
        origins: dict[str, list[str]] = defaultdict(list)

        def take(pairs: list[tuple[str, float]], origin: str) -> None:
            ids = []
            for cid, _ in pairs:
                ch = self.chunks.get(cid)
                if not ch:
                    continue
                # Soft language filter: keep matching language, plus a few cross-lingual
                ids.append(cid)
                origins[cid].append(origin)
            lists.append(ids)

        take(self._bm25(query, settings.sparse_top_k), "bm25")
        take(self._lsa(query, settings.dense_top_k), "lsa")

        # Strategy-specialized BM25: small children catch exact facts,
        # full passages catch broader descriptions.
        if query_type == "NUMERIC":
            take(self._bm25(query, 16, strategies={ChunkStrategy.PROPOSITION, ChunkStrategy.HIERARCHICAL_CHILD}), "bm25:prop")
        elif query_type in {"PERSON", "ENTITY", "LOCATION"}:
            take(self._bm25(query, 16, strategies={ChunkStrategy.HIERARCHICAL_CHILD, ChunkStrategy.SENTENCE_WINDOW}), "bm25:ent")
        else:
            take(self._bm25(query, 16, strategies={ChunkStrategy.PASSAGE, ChunkStrategy.SEMANTIC}), "bm25:pass")

        fused = rrf(lists)
        # Light language preference (not a hard filter)
        for cid, score in list(fused.items()):
            ch = self.chunks.get(cid)
            if ch and ch.language == language:
                fused[cid] = score * 1.08
            if ch and query_type in ch.query_types:
                fused[cid] = fused[cid] * 1.04

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

        # Parent expansion: collapse children onto unique parents, keep best child
        parent_best: dict[str, tuple[str, float]] = {}
        for cid, score in ranked:
            ch = self.chunks.get(cid)
            if not ch:
                continue
            prev = parent_best.get(ch.parent_id)
            if prev is None or score > prev[1]:
                parent_best[ch.parent_id] = (cid, score)

        hits: list[Hit] = []
        for rank, (pid, (cid, score)) in enumerate(
            sorted(parent_best.items(), key=lambda kv: kv[1][1], reverse=True)
        ):
            if rank >= settings.parent_expand:
                break
            ch = self.chunks[cid]
            hits.append(
                Hit(
                    chunk=ch,
                    score=float(score),
                    rank=rank,
                    origin="+".join(sorted(set(origins.get(cid, ["fused"])))),
                    parent_text=self.parents.get(pid, ch.text),
                )
            )
        return hits[:top_k]

    def _bm25(
        self,
        query: str,
        k: int,
        strategies: set[ChunkStrategy] | None = None,
    ) -> list[tuple[str, float]]:
        if self.bm25 is None:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        idxs, scores = self.bm25.retrieve(
            [q_tokens],
            k=min(k * 3, len(self.bm25_ids)),
            show_progress=False,
        )
        out: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], idxs[0], strict=False):
            cid = self.bm25_ids[int(idx)]
            ch = self.chunks.get(cid)
            if not ch:
                continue
            if strategies and ch.strategy not in strategies:
                continue
            out.append((cid, float(score)))
            if len(out) >= k:
                break
        return out

    def _lsa(self, query: str, k: int) -> list[tuple[str, float]]:
        if self.lsa_index is None:
            return []
        return self.lsa_index.search(self.lsa.encode_one(query), k)
