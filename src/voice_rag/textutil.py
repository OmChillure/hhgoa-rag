from __future__ import annotations

import hashlib
import re
import unicodedata

SENT_SPLIT = re.compile(r"(?<=[.!?।؟۔])\s+|(?<=\n)")
WORD_RE = re.compile(r"[A-Za-z0-9]+|[\u0900-\u097F]+", re.UNICODE)
STOP = frozenset(
    """
    a an the and or but if in on at to for of with from by as is are was were be
    been being it this that these those i you he she we they them my your our
    what which who whom how when where why not no do does did can could should
    would will just than then so such into over after before about
    """.split()
)
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
CLAUSE_SPLIT = re.compile(
    r"\s+(?:and then|but|however|although|because|which|where|while|;|:|—|–)\s+",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[\s\.\,;:\-\(\)\[\]\"]+", "", text).strip()
    return text


def content_hash(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()[:16]


def detect_language(text: str) -> str:
    if not text:
        return "en"
    dev = len(DEVANAGARI.findall(text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if dev > latin and dev >= 4:
        return "hi"
    return "en"


def sentences(text: str) -> list[str]:
    text = normalize(text)
    if not text:
        return []
    parts = [p.strip() for p in SENT_SPLIT.split(text) if p and p.strip()]
    if not parts:
        return [text]
    return parts


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text or "")]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOP and len(t) > 1]


def token_set(text: str) -> set[str]:
    return set(content_tokens(text))


def coverage(answer: str, context: str) -> float:
    ans = token_set(answer)
    if not ans:
        return 0.0
    ctx = token_set(context)
    if not ctx:
        return 0.0
    return len(ans & ctx) / len(ans)


def char_windows(text: str, size: int, overlap: int) -> list[tuple[int, int, str]]:
    text = normalize(text)
    if not text:
        return []
    if len(text) <= size:
        return [(0, len(text), text)]
    step = max(1, size - overlap)
    out: list[tuple[int, int, str]] = []
    i = 0
    while i < len(text):
        end = min(len(text), i + size)
        if end < len(text):
            # snap to nearest space so we don't cut mid-word
            snap = text.rfind(" ", i + size // 2, end)
            if snap > i:
                end = snap
        chunk = text[i:end].strip()
        if chunk:
            out.append((i, end, chunk))
        if end >= len(text):
            break
        i = max(i + 1, end - overlap)
        if i <= out[-1][0]:
            i = end
    return out


def infer_query_type(query: str) -> str:
    q = (query or "").lower()
    if re.search(r"\b(how many|how much|what year|when was|when did|population|age|number of|percentage|percent)\b", q):
        return "NUMERIC"
    if re.search(r"\b(where|which city|which country|located|capital of)\b", q):
        return "LOCATION"
    if re.search(r"\b(who is|who was|who were|which person|invented by|founded by)\b", q):
        return "PERSON"
    if re.search(r"\b(what is|what are|what does|define|meaning of|describe)\b", q):
        return "DESCRIPTION"
    if re.search(r"\b(which|who|what company|what organization)\b", q):
        return "ENTITY"
    return "UNKNOWN"
