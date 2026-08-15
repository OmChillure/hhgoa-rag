"""Multi-strategy chunking for MSMARCO-XI passages.

Ingest-time only. Query latency never re-splits the corpus.

Strategies (all run on every passage):

* passage          — keep the original MS MARCO passage intact (document-aware)
* sentence_window  — sliding sentence windows with overlap
* recursive        — separator cascade with character overlap
* semantic         — adjacent-sentence similarity breakpoints (TF-IDF cosine)
* proposition      — clause-level atomic facts
* hierarchical     — children (sentences) linked back to the parent passage

Overlap is explicit: sentence windows share a sentence, recursive windows share
characters, and every child stores prev/next context so retrieval can expand
without re-chunking at query time.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from voice_rag.textutil import (
    CLAUSE_SPLIT,
    char_windows,
    content_hash,
    normalize,
    sentences,
)
from voice_rag.types import Chunk, ChunkStrategy


def _base(
    *,
    parent_id: str,
    strategy: ChunkStrategy,
    text: str,
    language: str,
    query_types: list[str],
    source_query_ids: list[int],
    is_gold: bool,
    position: int,
    prev_ctx: str = "",
    next_ctx: str = "",
) -> Chunk:
    text = normalize(text)
    return Chunk(
        chunk_id=f"{strategy.value}:{parent_id}:{position}:{content_hash(text)}",
        parent_id=parent_id,
        strategy=strategy,
        text=text,
        language=language,
        query_types=query_types,
        source_query_ids=source_query_ids,
        is_gold=is_gold,
        position=position,
        n_chars=len(text),
        prev_ctx=prev_ctx,
        next_ctx=next_ctx,
    )


def chunk_passage(text: str, parent_id: str, language: str, **meta) -> list[Chunk]:
    text = normalize(text)
    if not text:
        return []
    return [
        _base(
            parent_id=parent_id,
            strategy=ChunkStrategy.PASSAGE,
            text=text,
            language=language,
            position=0,
            **meta,
        )
    ]


def chunk_sentence_window(
    text: str,
    parent_id: str,
    language: str,
    window: int = 2,
    stride: int = 1,
    **meta,
) -> list[Chunk]:
    sents = sentences(text)
    if not sents:
        return []
    if len(sents) <= window:
        return [
            _base(
                parent_id=parent_id,
                strategy=ChunkStrategy.SENTENCE_WINDOW,
                text=" ".join(sents),
                language=language,
                position=0,
                **meta,
            )
        ]
    out: list[Chunk] = []
    pos = 0
    for i in range(0, len(sents), stride):
        piece = sents[i : i + window]
        if not piece:
            break
        prev_ctx = sents[i - 1] if i > 0 else ""
        nxt = sents[i + window] if i + window < len(sents) else ""
        out.append(
            _base(
                parent_id=parent_id,
                strategy=ChunkStrategy.SENTENCE_WINDOW,
                text=" ".join(piece),
                language=language,
                position=pos,
                prev_ctx=prev_ctx,
                next_ctx=nxt,
                **meta,
            )
        )
        pos += 1
        if i + window >= len(sents):
            break
    return out


def chunk_recursive(
    text: str,
    parent_id: str,
    language: str,
    size: int = 220,
    overlap: int = 50,
    **meta,
) -> list[Chunk]:
    windows = char_windows(text, size=size, overlap=overlap)
    return [
        _base(
            parent_id=parent_id,
            strategy=ChunkStrategy.RECURSIVE,
            text=chunk,
            language=language,
            position=i,
            **meta,
        )
        for i, (_, _, chunk) in enumerate(windows)
    ]


def chunk_propositions(text: str, parent_id: str, language: str, **meta) -> list[Chunk]:
    sents = sentences(text)
    props: list[str] = []
    for sent in sents:
        parts = [p.strip(" ,;") for p in CLAUSE_SPLIT.split(sent) if p and p.strip()]
        if len(parts) <= 1:
            props.append(sent)
        else:
            props.extend(p for p in parts if len(p) > 20)
    if not props:
        return []
    out: list[Chunk] = []
    for i, prop in enumerate(props):
        prev_ctx = props[i - 1] if i else ""
        next_ctx = props[i + 1] if i + 1 < len(props) else ""
        out.append(
            _base(
                parent_id=parent_id,
                strategy=ChunkStrategy.PROPOSITION,
                text=prop,
                language=language,
                position=i,
                prev_ctx=prev_ctx,
                next_ctx=next_ctx,
                **meta,
            )
        )
    return out


def chunk_hierarchical_children(text: str, parent_id: str, language: str, **meta) -> list[Chunk]:
    sents = sentences(text)
    out: list[Chunk] = []
    for i, sent in enumerate(sents):
        if len(sent) < 20:
            continue
        out.append(
            _base(
                parent_id=parent_id,
                strategy=ChunkStrategy.HIERARCHICAL_CHILD,
                text=sent,
                language=language,
                position=i,
                prev_ctx=sents[i - 1] if i else "",
                next_ctx=sents[i + 1] if i + 1 < len(sents) else "",
                **meta,
            )
        )
    return out


def chunk_semantic(
    text: str,
    parent_id: str,
    language: str,
    *,
    vectorizer: TfidfVectorizer | None = None,
    **meta,
) -> list[Chunk]:
    """Split where adjacent-sentence cosine similarity drops below a local valley."""
    sents = [s for s in sentences(text) if len(s) > 15]
    if len(sents) <= 2:
        return [
            _base(
                parent_id=parent_id,
                strategy=ChunkStrategy.SEMANTIC,
                text=normalize(text),
                language=language,
                position=0,
                **meta,
            )
        ]
    try:
        if vectorizer is not None:
            mat = vectorizer.transform(sents)
        else:
            local = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            mat = local.fit_transform(sents)
        sims = []
        for i in range(len(sents) - 1):
            sims.append(float(cosine_similarity(mat[i], mat[i + 1])[0, 0]))
        if not sims:
            threshold = 0.0
        else:
            arr = np.array(sims, dtype=np.float32)
            threshold = float(arr.mean() - 0.35 * arr.std()) if arr.std() > 1e-6 else float(arr.mean() * 0.7)
        groups: list[list[str]] = [[sents[0]]]
        for i, sim in enumerate(sims):
            if sim < threshold and len(groups[-1]) >= 1:
                groups.append([sents[i + 1]])
            else:
                groups[-1].append(sents[i + 1])
    except ValueError:
        groups = [sents]
    out: list[Chunk] = []
    for i, group in enumerate(groups):
        out.append(
            _base(
                parent_id=parent_id,
                strategy=ChunkStrategy.SEMANTIC,
                text=" ".join(group),
                language=language,
                position=i,
                **meta,
            )
        )
    return out


class EnsembleChunker:
    """Run every strategy and drop exact-duplicate texts within a parent."""

    def __init__(self) -> None:
        self._tfidf: TfidfVectorizer | None = None

    def fit_semantic(self, corpus: list[str]) -> None:
        sample = [c for c in corpus if c][:8000]
        if not sample:
            return
        self._tfidf = TfidfVectorizer(
            max_features=12000,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self._tfidf.fit(sample)

    def chunk_one(
        self,
        text: str,
        parent_id: str,
        language: str,
        query_types: list[str] | None = None,
        source_query_ids: list[int] | None = None,
        is_gold: bool = False,
    ) -> list[Chunk]:
        meta = {
            "query_types": query_types or [],
            "source_query_ids": source_query_ids or [],
            "is_gold": is_gold,
        }
        raw: list[Chunk] = []
        raw += chunk_passage(text, parent_id, language, **meta)
        raw += chunk_sentence_window(text, parent_id, language, window=2, stride=1, **meta)
        # Full family on English. Hindi keeps passage + overlapping
        # sentence windows so the Indic BM25 index stays useful without
        # exploding the chunk count (MSMARCO-XI duplicates every passage).
        if language == "en":
            raw += chunk_recursive(text, parent_id, language, **meta)
            raw += chunk_semantic(text, parent_id, language, vectorizer=self._tfidf, **meta)
            raw += chunk_propositions(text, parent_id, language, **meta)
            raw += chunk_hierarchical_children(text, parent_id, language, **meta)

        seen: set[tuple[str, str]] = set()
        out: list[Chunk] = []
        for ch in raw:
            key = (ch.strategy.value, content_hash(ch.text))
            if key in seen or len(ch.text) < 12:
                continue
            seen.add(key)
            out.append(ch)
        return out

    def chunk_many(self, records: list[dict]) -> tuple[list[Chunk], dict[str, str]]:
        """records: parent_id, text, language, query_types, source_query_ids, is_gold."""
        self.fit_semantic([r["text"] for r in records])
        chunks: list[Chunk] = []
        parents: dict[str, str] = {}
        for rec in records:
            pid = rec["parent_id"]
            parents[pid] = rec["text"]
            chunks.extend(
                self.chunk_one(
                    rec["text"],
                    pid,
                    rec["language"],
                    query_types=rec.get("query_types") or [],
                    source_query_ids=rec.get("source_query_ids") or [],
                    is_gold=bool(rec.get("is_gold")),
                )
            )
        return chunks, parents

    @staticmethod
    def strategy_counts(chunks: list[Chunk]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for c in chunks:
            counts[c.strategy.value] += 1
        return dict(counts)
