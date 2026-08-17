from __future__ import annotations

import hashlib
import re
import unicodedata

SENT_SPLIT = re.compile(r"(?<=[.!?।؟۔])\s+|(?<=\n)")
WORD_RE = re.compile(
    r"[A-Za-z0-9]+"
    r"|[\u0900-\u097F]+"
    r"|[\u0980-\u09FF]+"
    r"|[\u0A00-\u0A7F]+"
    r"|[\u0A80-\u0AFF]+"
    r"|[\u0B00-\u0B7F]+"
    r"|[\u0B80-\u0BFF]+"
    r"|[\u0C00-\u0C7F]+"
    r"|[\u0C80-\u0CFF]+"
    r"|[\u0D00-\u0D7F]+"
    r"|[\u0600-\u06FF]+",
    re.UNICODE,
)
STOP = frozenset(
    """
    a an the and or but if in on at to for of with from by as is are was were be
    been being it this that these those i you he she we they them my your our
    what which who whom how when where why not no do does did can could should
    would will just than then so such into over after before about
    """.split()
)
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_SCRIPT_LANG: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[\u0980-\u09FF]"), "bn"),  # also covers Assamese
    (re.compile(r"[\u0A00-\u0A7F]"), "pa"),
    (re.compile(r"[\u0A80-\u0AFF]"), "gu"),
    (re.compile(r"[\u0B00-\u0B7F]"), "or"),
    (re.compile(r"[\u0B80-\u0BFF]"), "ta"),
    (re.compile(r"[\u0C00-\u0C7F]"), "te"),
    (re.compile(r"[\u0C80-\u0CFF]"), "kn"),
    (re.compile(r"[\u0D00-\u0D7F]"), "ml"),
    (re.compile(r"[\u0600-\u06FF]"), "ur"),
]
CLAUSE_SPLIT = re.compile(
    r"\s+(?:and then|but|however|although|because|which|where|while|;|:|—|–)\s+",
    re.IGNORECASE,
)

# Shared-script groups. Used only as a last-resort shard fallback; the
# detector below tries to pick the exact language from distinctive tokens.
SCRIPT_AMBIGUITY_GROUPS: dict[str, tuple[str, ...]] = {
    "hi": ("hi", "mr", "sa", "ne"),
    "bn": ("bn", "as"),
}

# Devanagari / Eastern-Nagari disambiguation. Cheap lexical cues beat
# searching every sibling BM25 shard (~4x retrieve cost).
# Indic matras are not \w, so \b is unreliable. Match the tokens directly.
_DEV_MARATHI = re.compile(r"ळ|[अआ]हे|कोणत|च्या|आणि")
_DEV_NEPALI = re.compile(r"के हो|को हो| छ[?।]| हो[?।]| हो$")
_DEV_SANSKRIT = re.compile(r"(?:^|\s)(?:अस्ति|भवति|किम्)(?:\s|$|[?।])")
_AS_LETTERS = re.compile(r"[ৰৱ]")


def language_shard_candidates(language: str) -> tuple[str, ...]:
    return SCRIPT_AMBIGUITY_GROUPS.get(language, (language,))


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
    latin = len(re.findall(r"[A-Za-z]", text))
    best_lang = "en"
    best_count = latin
    for pattern, lang in _SCRIPT_LANG:
        count = len(pattern.findall(text))
        if count > best_count:
            best_count = count
            best_lang = lang
    dev = len(DEVANAGARI.findall(text))
    if dev > best_count and dev >= 4:
        if _DEV_MARATHI.search(text):
            return "mr"
        if _DEV_NEPALI.search(text):
            return "ne"
        if _DEV_SANSKRIT.search(text):
            return "sa"
        return "hi"
    if best_lang != "en" and best_count >= 4:
        if best_lang == "bn" and _AS_LETTERS.search(text):
            return "as"
        return best_lang
    return "en"


_ABBR = re.compile(r"^(?:डॉ|श्री|प्रो|smt|mr|mrs|dr|ms|prof)\.?$", re.IGNORECASE)
# Hide abbreviation dots so "डॉ. राजेंद्र" is not treated as a sentence end.
_ABBR_DOT = re.compile(r"(डॉ|श्री|प्रो|Smt|Mr|Mrs|Ms|Dr|Prof)\.", re.IGNORECASE)


def sentences(text: str) -> list[str]:
    text = normalize(text)
    if not text:
        return []
    marked = _ABBR_DOT.sub(lambda m: m.group(1) + "\u2024", text)
    parts = [p.replace("\u2024", ".").strip() for p in SENT_SPLIT.split(marked) if p and p.strip()]
    if not parts:
        return [text]
    merged: list[str] = []
    for part in parts:
        if merged and _ABBR.match(merged[-1].rstrip(".")):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


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
    if re.search(
        r"\b(how many|how much|what year|when was|when did|population|age|number of|percentage|percent)\b"
        r"|जनसंख्या|कितने|कितनी|সংখ্যা|எத்தனை|વસ્તી|ਆਬਾਦੀ|آبادی",
        q,
    ):
        return "NUMERIC"
    if re.search(
        r"\b(where|which city|which country|located|capital of)\b"
        r"|राजधानी|राजधानी|தலைநகரம்|રાજધાની|ਰਾਜਧਾਨੀ|دارالحکومت|തലസ്ഥാനം|ರಾಜಧಾನಿ|ରାଜଧାନୀ|রাজধানী|ৰাজধানী",
        q,
    ):
        return "LOCATION"
    if re.search(
        r"\b(who is|who was|who were|who invented|which person|invented by|founded by)\b"
        r"|किसने|कौन थे|कौन है|कौन था|कोण होते|को थिए|யார்|કોણ|ਕੌਣ|کون |ആര്|ಯಾರು|କିଏ|কে ছিলেন|কোন আছিল",
        q,
    ):
        return "PERSON"
    if re.search(
        r"\b(what is|what are|what does|define|meaning of|describe)\b"
        r"|क्या है|क्या होता|என்ன|શું છે|ਕੀ ਹੈ|کیا ہے|എന്താണ്|ಯಾವುದು|କଣ|কী\?| কি\?",
        q,
    ):
        return "DESCRIPTION"
    if re.search(r"\b(which|who|what company|what organization)\b", q):
        return "ENTITY"
    return "UNKNOWN"
