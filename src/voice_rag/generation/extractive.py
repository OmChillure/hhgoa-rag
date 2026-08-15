"""Sub-10ms extractive reader.

MS MARCO answers are short spans. We score candidate sentences / clauses
from retrieved *parent* passages (query-time span chunking) with a linear
combination of lexical overlap, exact phrase hits, and type-aware bonuses.
No model call on the SLA path.
"""

from __future__ import annotations

import re

from voice_rag.textutil import CLAUSE_SPLIT, content_tokens, sentences
from voice_rag.types import Hit, Span


_NUM = re.compile(r"\d[\d,.]*%?")


def _candidates(text: str) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    cursor = 0
    for sent in sentences(text):
        start = text.find(sent, cursor)
        if start < 0:
            start = cursor
        end = start + len(sent)
        out.append((start, end, sent))
        for part in CLAUSE_SPLIT.split(sent):
            part = part.strip(" ,;")
            if 20 < len(part) < len(sent):
                ps = text.find(part, start)
                if ps >= 0:
                    out.append((ps, ps + len(part), part))
        cursor = end
    return out


def _score(query: str, cand: str, query_type: str) -> float:
    q_toks = content_tokens(query)
    c_toks = content_tokens(cand)
    if not q_toks or not c_toks:
        return 0.0
    q_set, c_set = set(q_toks), set(c_toks)
    overlap = len(q_set & c_set) / (len(q_set) ** 0.5)
    # phrase bonus
    q_low, c_low = query.lower(), cand.lower()
    phrase = 0.0
    for n in (4, 3, 2):
        grams = [" ".join(q_toks[i : i + n]) for i in range(len(q_toks) - n + 1)]
        hits = sum(1 for g in grams if g in c_low)
        if hits:
            phrase += hits * n * 0.15
            break
    type_bonus = 0.0
    if query_type == "NUMERIC":
        q_nums = set(_NUM.findall(query))
        c_nums = _NUM.findall(cand)
        if c_nums:
            type_bonus += 0.45
        if q_nums and any(n in cand for n in q_nums):
            type_bonus += 0.2
    if query_type in {"PERSON", "ENTITY", "LOCATION"}:
        caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", cand)
        if caps:
            type_bonus += 0.2
    length_pen = 0.0 if 40 <= len(cand) <= 280 else -0.15
    return overlap + phrase + type_bonus + length_pen


def extract(query: str, hits: list[Hit], query_type: str = "UNKNOWN") -> tuple[str, list[Span], float]:
    scored: list[tuple[float, Span, str]] = []
    for hit in hits:
        parent = hit.parent_text or hit.chunk.text
        for start, end, cand in _candidates(parent):
            s = _score(query, cand, query_type) + 0.35 * hit.score
            scored.append(
                (
                    s,
                    Span(text=cand, parent_id=hit.chunk.parent_id, start=start, end=end, score=s),
                    parent,
                )
            )
    if not scored:
        return "", [], 0.0
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_span, _ = scored[0]
    # optionally stitch a neighboring high-scoring span from the same parent
    extras = [
        s
        for sc, s, _ in scored[1:4]
        if s.parent_id == best_span.parent_id and s.text != best_span.text and sc > best_score * 0.72
    ]
    answer = best_span.text
    spans = [best_span]
    if extras:
        nxt = extras[0]
        if nxt.text not in answer:
            if nxt.start >= best_span.end:
                answer = f"{answer} {nxt.text}"
            else:
                answer = f"{nxt.text} {answer}"
            spans.append(nxt)
    # confidence: squash score into 0-1
    conf = max(0.0, min(1.0, best_score / 3.2))
    return answer.strip(), spans, conf
