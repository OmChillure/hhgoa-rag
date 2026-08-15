"""Input / output guardrails.

The system is allowed to refuse. Refusals are first-class outcomes, not errors.
"""

from __future__ import annotations

import re

from voice_rag.config import settings
from voice_rag.textutil import coverage, token_set, tokenize
from voice_rag.types import GuardDecision, Hit


_UNSAFE = [
    (r"\b(how to make|build|synthesize)\b.*\b(bomb|explosive|napalm|ricin|fentanyl|sarin)\b", "weapons"),
    (r"\b(kill|murder|assassinate|poison)\b.*\b(someone|him|her|people|kids)\b", "violence"),
    (r"\b(suicide|self[- ]harm|kill myself)\b", "self_harm"),
    (r"\b(child porn|csam|underage sex|sexual(?:ly)? (?:explicit )?(?:content )?of (?:a )?minor)\b", "csea"),
    (r"\b(credit card|ssn|social security|cvv)\b.*\b(dump|steal|generate)\b", "crime"),
    (r"\b(ignore (?:all )?(?:previous|prior) (?:instructions|rules)|jailbreak|dan mode)\b", "jailbreak"),
]

_OFF_TOPIC = [
    r"\b(write (?:me )?(?:a )?.{0,20}(?:poem|song|essay|cover letter))\b",
    r"\b(act as|you are now|roleplay as)\b",
    r"\b(generate (?:python|javascript|code)|write a function|leetcode)\b",
    r"\b(what do you think of me|are you single|tell me a joke)\b",
    r"\b(who will win|stock pick|invest in)\b",
]


def check_input(query: str) -> GuardDecision:
    q = (query or "").strip()
    if not q:
        return GuardDecision(allowed=False, stage="input", reason="empty_query", categories=["empty"])
    if len(q) > settings.max_query_chars:
        return GuardDecision(allowed=False, stage="input", reason="too_long", categories=["length"])
    if len(tokenize(q)) < 2 and not q.endswith("?"):
        return GuardDecision(
            allowed=False,
            stage="input",
            reason="too_short",
            categories=["thin_query"],
        )
    low = q.lower()
    cats = []
    for pat, cat in _UNSAFE:
        if re.search(pat, low):
            cats.append(cat)
    if cats:
        return GuardDecision(
            allowed=False,
            stage="input",
            reason="unsafe_query",
            categories=cats,
        )
    off = [pat for pat in _OFF_TOPIC if re.search(pat, low)]
    if off:
        return GuardDecision(
            allowed=False,
            stage="input",
            reason="off_topic",
            categories=["off_topic"],
            details={"patterns": off[:3]},
        )
    return GuardDecision(allowed=True, stage="input", reason="ok")


def check_retrieval(hits: list[Hit], query: str = "") -> GuardDecision:
    if not hits:
        return GuardDecision(
            allowed=False,
            stage="retrieval",
            reason="no_hits",
            categories=["unanswerable"],
        )
    best = hits[0].score
    qset = token_set(query)
    ctx = token_set(" ".join((h.parent_text or h.chunk.text) for h in hits[:4]))
    support = (len(qset & ctx) / len(qset)) if qset else 0.0
    if best < settings.min_retrieval_score or support < 0.55:
        return GuardDecision(
            allowed=False,
            stage="retrieval",
            reason="low_confidence",
            categories=["unanswerable", "off_corpus"],
            details={"best_score": best, "query_support": support},
        )
    return GuardDecision(
        allowed=True,
        stage="retrieval",
        reason="ok",
        details={"best_score": best, "k": len(hits), "query_support": support},
    )


def check_grounding(answer: str, hits: list[Hit]) -> GuardDecision:
    if not answer:
        return GuardDecision(
            allowed=False,
            stage="grounding",
            reason="empty_answer",
            categories=["unanswerable"],
        )
    ctx = " ".join(h.parent_text or h.chunk.text for h in hits)
    cov = coverage(answer, ctx)
    details = {"coverage": cov, "threshold": settings.grounding_min_coverage}
    if cov < settings.grounding_min_coverage:
        return GuardDecision(
            allowed=False,
            stage="grounding",
            reason="ungrounded",
            categories=["hallucination"],
            details=details,
        )
    return GuardDecision(allowed=True, stage="grounding", reason="ok", details=details)


REFUSALS = {
    "empty_query": "I need a question to look up.",
    "too_short": "That query is too thin to retrieve against. Ask a full question.",
    "too_long": "Please shorten the question.",
    "unsafe_query": "I can't help with that request.",
    "off_topic": "I'm a grounded QA system over MSMARCO-XI. Ask a factual question I can retrieve.",
    "no_hits": "I don't have a passage that matches that question.",
    "low_confidence": "Nothing in the indexed MSMARCO-XI subset is close enough to answer reliably.",
    "empty_answer": "I couldn't extract a grounded answer from the retrieved passages.",
    "ungrounded": "I found related passages, but I won't invent an answer that isn't supported by them.",
}


def refusal_text(reason: str) -> str:
    return REFUSALS.get(reason, "I can't answer that from the retrieved context.")
