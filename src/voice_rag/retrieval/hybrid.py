from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from voice_rag.config import settings
from voice_rag.retrieval.store import FaissIndex
from voice_rag.textutil import (
    bm25_query_tokens,
    capital_alignment,
    content_tokens,
    definition_alignment,
    detect_language,
    infer_query_type,
    language_shard_candidates,
    looks_like_question,
    tokenize,
)
from voice_rag.types import Chunk, ChunkStrategy, Hit

# Question-side verbs/stems that an *answer* passage is allowed to omit.
_Q_STEMS = frozenset(
    """
    invented founded created called named located meaning definition define
    describe explain happen happened caused started become used using
    """.split()
)
# Structural question words. Missing these is weaker evidence than missing
# a topical noun like "phloem" or "odisha".
_STRUCTURE = frozenset(
    """
    direction number amount people country first largest current meaning
    difference types type world years year after before during
    """.split()
)

_CHUNK_CACHE_MAX = 50_000


def rrf(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranks in rank_lists:
        for i, cid in enumerate(ranks):
            scores[cid] += 1.0 / (k + i + 1)
    return dict(scores)


def specific_terms(query: str) -> list[str]:
    """Content tokens that should appear in a relevant passage."""
    return [t for t in content_tokens(query) if len(t) >= 5 and t not in _Q_STEMS]


def lexical_relevance(query: str, text: str) -> float:
    """Cheap passage score. Typical range roughly 0..3."""
    q_toks = content_tokens(query)
    if not q_toks:
        return 0.0
    t_toks = content_tokens(text)
    t_set = set(t_toks)
    q_set = set(q_toks)
    cover = len(q_set & t_set) / len(q_set)
    spec = specific_terms(query)
    spec_hit = (sum(1 for t in spec if t in t_set) / len(spec)) if spec else cover
    t_low = text.lower()
    phrase = 0.0
    for n in (4, 3, 2):
        grams = [" ".join(q_toks[i : i + n]) for i in range(len(q_toks) - n + 1)]
        if grams and any(g in t_low for g in grams):
            phrase = n / 4.0
            break
    # Keep stopwords for phrases like "population of india" / "capital of france".
    q_all = tokenize(query)
    while q_all and q_all[0] in {"what", "who", "which", "where", "when", "how", "is", "was", "are", "were", "the", "a", "an", "does", "do", "did"}:
        q_all = q_all[1:]
    q_phrase = " ".join(q_all)
    if q_phrase and q_phrase in t_low:
        phrase = max(phrase, 0.85)
    echo = 0.0
    q_low = query.lower().rstrip("?").strip()
    head = t_low.lstrip()
    if head.startswith("question:") or (q_low and head.startswith(q_low[: min(40, len(q_low))])):
        echo = 0.55
    topic = [t for t in spec if t not in _STRUCTURE]
    topic_pen = 0.0
    if topic:
        topic_pen = 0.85 * (1.0 - sum(1 for t in topic if t in t_set) / len(topic))
    defn = 0.0
    if re.search(r"\b(?:is|was|are|were) the\b", t_low) and not head.startswith(("what ", "who ", "which ", "where ")):
        defn = 0.35
    defn += definition_alignment(text, query)
    cap = 1.15 * capital_alignment(query, text)
    qpen = 0.85 if looks_like_question(text) else 0.0
    return 1.5 * spec_hit + 0.75 * cover + 0.7 * phrase + defn + cap - echo - topic_pen - qpen


def has_all_specific(query: str, text: str) -> bool:
    spec = specific_terms(query)
    if not spec:
        return False
    t_set = set(content_tokens(text))
    return all(t in t_set for t in spec)


class HybridRetriever:
    """Multi-index hybrid search over the sqlite-backed passage index.

    BM25 (sharded per language) + sentence-transformer/FAISS dense search,
    fused with RRF on a single id space (passage pid), then lexically reranked.
    """

    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}
        self.lsa_index: FaissIndex | None = None
        # language -> (bm25s.BM25, doc_ids). Sharding per language (see
        # scripts/build_bm25.py) keeps each query's dense relevance-array
        # scan to one shard (~550K docs) instead of the full corpus
        # (~7.77M docs), which is what makes bm25s fast enough for the
        # latency budget.
        self.bm25_shards: dict[str, tuple[Any, list[str]]] = {}
        self.st_model = None
        self._db: sqlite3.Connection | None = None

    def load(self, directory: Path) -> None:
        db_path = directory / "passages.db"
        self._db = sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA cache_size=-131072")
        self._db.execute("PRAGMA mmap_size=1073741824")
        self._db.execute("PRAGMA temp_store=MEMORY")
        n = self._db.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
        print(f"sqlite passages: {n}", flush=True)

        meta = json.loads((directory / "encoder.meta.json").read_text(encoding="utf-8"))
        from sentence_transformers import SentenceTransformer

        self.st_model = SentenceTransformer(meta["model"])
        nprobe = int(getattr(settings, "faiss_nprobe", None) or meta.get("nprobe") or 8)
        self.lsa_index = FaissIndex.load(
            directory / "lsa",
            int(meta.get("dim") or 384),
            nprobe=nprobe,
        )

        import bm25s

        bm25_dir = directory / "bm25"
        for shard_dir in bm25_dir.glob("*"):
            if not shard_dir.is_dir():
                continue
            lang = shard_dir.name
            try:
                shard_bm25 = bm25s.BM25.load(
                    str(shard_dir), load_vocab=True, mmap=True, backend="numba"
                )
            except Exception:
                shard_bm25 = bm25s.BM25.load(str(shard_dir), load_vocab=True, mmap=True)
            shard_ids = json.loads((shard_dir / "doc_ids.json").read_text(encoding="utf-8"))
            self.bm25_shards[lang] = (shard_bm25, shard_ids)
        print(f"bm25 shards: {sorted(self.bm25_shards)}", flush=True)
        self._warmup_bm25()

    def stats(self) -> dict:
        n = int(self._db.execute("SELECT COUNT(*) FROM passages").fetchone()[0])
        return {
            "chunks": n,
            "parents": n,
            "lsa_dim": self.lsa_index.dim if self.lsa_index else 0,
            "encoder": "st" if self.st_model is not None else "lsa",
            "sqlite": True,
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
        pool_k = max(top_k, int(getattr(settings, "rerank_pool", 48) or 48))

        lists: list[list[str]] = []
        origins: dict[str, list[str]] = defaultdict(list)

        def take(pairs: list[tuple[str, float]], origin: str) -> None:
            if not pairs:
                return
            hydrated = self._hydrate([cid for cid, _ in pairs])
            ids: list[str] = []
            seen: set[str] = set()
            for cid, _ in pairs:
                ch = hydrated.get(cid) or self.get_chunk(cid)
                if ch is None:
                    continue
                pid = ch.chunk_id
                if pid in seen:
                    continue
                seen.add(pid)
                ids.append(pid)
                origins[pid].append(origin)
            if ids:
                lists.append(ids)

        # One shard only. Extra sibling shards (hi→mr/sa/ne/en) were the
        # 100ms+ tail; numba BM25 on the right shard is ~1-5ms.
        primary = self._shard_langs(language, expand=False)
        take(self._bm25_from(query, settings.sparse_top_k, primary), "bm25")

        if not lists and language != "en" and "en" in self.bm25_shards:
            take(self._bm25_from(query, settings.sparse_top_k, ["en"]), "bm25:en")

        if query_type == "DESCRIPTION":
            heads = [t for t in specific_terms(query) if t not in {"meaning", "define", "describe"}]
            if heads:
                take(self._bm25_from(" ".join(heads[:2]), 24, primary), "bm25:head")

        if self._needs_dense(query, query_type, lists[0] if lists else []):
            take(self._dense(query, settings.dense_top_k), "dense")

        fused = rrf(lists)
        for cid, score in list(fused.items()):
            ch = self.get_chunk(cid)
            if ch and ch.language == language:
                fused[cid] = score * 1.10
            if query_type == "NUMERIC" and ch and any(c.isdigit() for c in ch.text):
                fused[cid] = fused[cid] * 1.04

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:pool_k]
        ranked = self._rerank(query, ranked, language)

        hits: list[Hit] = []
        for rank, (cid, score) in enumerate(ranked):
            if rank >= top_k:
                break
            ch = self.get_chunk(cid)
            if not ch:
                continue
            hits.append(
                Hit(
                    chunk=ch,
                    score=float(score),
                    rank=rank,
                    origin="+".join(sorted(set(origins.get(cid, ["fused"])))),
                    parent_text=ch.text,
                )
            )
        return hits[:top_k]

    def _rerank(
        self,
        query: str,
        ranked: list[tuple[str, float]],
        language: str,
    ) -> list[tuple[str, float]]:
        if len(ranked) <= 1:
            return ranked
        rrf_scores = [s for _, s in ranked]
        rmin, rmax = min(rrf_scores), max(rrf_scores)
        span = (rmax - rmin) or 1e-9
        scored: list[tuple[float, str, float]] = []
        for pid, rrf_s in ranked:
            ch = self.get_chunk(pid)
            if ch is None:
                continue
            lex = lexical_relevance(query, ch.text)
            rrf_n = (rrf_s - rmin) / span
            lang_b = 0.10 if ch.language == language else 0.0
            final = 0.40 * rrf_n + 0.55 * (lex / 2.9) + lang_b
            scored.append((final, pid, rrf_s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(pid, final) for final, pid, _ in scored]

    def _warmup_bm25(self) -> None:
        """Trigger numba JIT so the first user query isn't the compile hit."""
        for bm25, _ids in self.bm25_shards.values():
            vocab = getattr(bm25, "vocab_dict", None) or {}
            tok = next(iter(vocab), None)
            if not tok:
                continue
            try:
                bm25.retrieve([[tok]], k=1, show_progress=False)
            except Exception:
                continue

    def _needs_dense(self, query: str, query_type: str, ids: list[str]) -> bool:
        """Dense is the semantic backstop. Skip it only when BM25 already looks right."""
        if not ids:
            return True
        if query_type == "DESCRIPTION":
            for pid in ids[:4]:
                ch = self.get_chunk(pid)
                if ch is not None and definition_alignment(ch.text, query) > 0:
                    return False
            return True
        if query_type == "LOCATION":
            ch = self.get_chunk(ids[0])
            return ch is None or capital_alignment(query, ch.text) <= 0
        return not self._strong_bm25(query, ids)

    def _strong_bm25(self, query: str, ids: list[str]) -> bool:
        """True when lexical retrieval already covers the query well enough to skip dense."""
        if not ids:
            return False
        spec = specific_terms(query)
        if not spec:
            return True
        need = 1 if len(spec) == 1 else max(1, (len(spec) + 1) // 2)
        for pid in ids[:6]:
            ch = self.get_chunk(pid)
            if ch is None:
                continue
            t_set = set(content_tokens(ch.text))
            if sum(1 for t in spec if t in t_set) >= need:
                return True
        return False

    def _shard_langs(self, language: str, *, expand: bool) -> list[str]:
        langs: list[str] = []
        if language in self.bm25_shards:
            langs.append(language)
        elif expand:
            for cand in language_shard_candidates(language):
                if cand in self.bm25_shards and cand not in langs:
                    langs.append(cand)
        if expand or not langs:
            for cand in language_shard_candidates(language):
                if cand in self.bm25_shards and cand not in langs:
                    langs.append(cand)
            if "en" not in langs and "en" in self.bm25_shards:
                langs.append("en")
        return langs

    def get_chunk(self, cid: str) -> Chunk | None:
        if cid in self.chunks:
            return self.chunks[cid]
        hydrated = self._hydrate([cid])
        return hydrated.get(cid)

    def _chunk_from_row(self, row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["pid"],
            parent_id=row["pid"],
            strategy=ChunkStrategy.PASSAGE,
            text=row["text"],
            language=row["language"] or "en",
        )

    def _remember(self, row: sqlite3.Row) -> Chunk:
        ch = self._chunk_from_row(row)
        if len(self.chunks) >= _CHUNK_CACHE_MAX:
            self.chunks.clear()
        self.chunks[ch.chunk_id] = ch
        self.chunks[str(row["rowid"])] = ch
        return ch

    def _hydrate(self, ids: list[str]) -> dict[str, Chunk]:
        out: dict[str, Chunk] = {}
        missing: list[str] = []
        for cid in ids:
            ch = self.chunks.get(cid)
            if ch is not None:
                out[cid] = ch
            else:
                missing.append(cid)
        if not missing or self._db is None:
            return out

        pids = [c for c in missing if not c.isdigit()]
        rowids = [int(c) for c in missing if c.isdigit()]
        if pids:
            q = f"SELECT rowid, pid, text, language FROM passages WHERE pid IN ({','.join('?' * len(pids))})"
            for row in self._db.execute(q, pids):
                ch = self._remember(row)
                out[ch.chunk_id] = ch
        if rowids:
            q = f"SELECT rowid, pid, text, language FROM passages WHERE rowid IN ({','.join('?' * len(rowids))})"
            for row in self._db.execute(q, rowids):
                ch = self._remember(row)
                out[str(row["rowid"])] = ch
                out[ch.chunk_id] = ch
        return out

    def _bm25(
        self,
        query: str,
        k: int,
        language: str | None = None,
    ) -> list[tuple[str, float]]:
        langs = self._shard_langs(language or "en", expand=True)
        return self._bm25_from(query, k, langs)

    def _bm25_from(self, query: str, k: int, langs: list[str]) -> list[tuple[str, float]]:
        q_tokens = bm25_query_tokens(query)
        if not q_tokens or not langs:
            return []
        # Per-shard lists are RRF'd separately by the caller. Here we keep
        # encounter order (primary shard first) and drop raw-score merges
        # across shards — those scores are not comparable.
        out: list[tuple[str, float]] = []
        seen: set[str] = set()
        for lang in langs:
            shard = self.bm25_shards.get(lang)
            if not shard:
                continue
            shard_bm25, shard_ids = shard
            for cid, score in self._bm25_search(shard_bm25, shard_ids, q_tokens, k):
                if cid in seen:
                    continue
                seen.add(cid)
                out.append((cid, score))
        return out

    @staticmethod
    def _bm25_search(bm25: Any, doc_ids: list[str], q_tokens: list[str], k: int) -> list[tuple[str, float]]:
        take = min(max(k, 1), len(doc_ids))
        # Match the shard backend: numba indexes require backend_selection=numba
        # (or auto). Forcing numpy here 500s the API.
        backend = getattr(bm25, "backend", "numpy")
        idxs, scores = bm25.retrieve(
            [q_tokens],
            k=take,
            show_progress=False,
            backend_selection="numba" if backend == "numba" else "numpy",
        )
        return [
            (doc_ids[int(idx)], float(score))
            for score, idx in zip(scores[0], idxs[0], strict=False)
            if int(idx) >= 0
        ]

    def _dense(self, query: str, k: int) -> list[tuple[str, float]]:
        if self.lsa_index is None or self.st_model is None:
            return []
        vec = self.st_model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return self.lsa_index.search(vec[0], k)
