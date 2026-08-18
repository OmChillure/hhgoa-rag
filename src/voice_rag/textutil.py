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


# Words that name the relation "capital", not the place being asked about.
_CAPITAL_LEX = frozenset(
    """
    capital राजधानी तலைநகரம் રાજધાની ਰਾਜਧਾਨੀ دارالحکومت
    തലസ്ഥാനം ರಾಜಧಾನಿ ରାଜଧାନୀ রাজধানী ৰাজধানী
    """.split()
)
_SUBJECT_STOP = frozenset(
    """
    क्या कौन कहाँ कहां है था थे की का के को कौनसी कौनसा
    कोणती आहे च्या ची चा चे क्याहे
    क्या है எது என்ன கை કઈ કે ਕੀ کیا
    কোথায় কী কি ಯಾವುದು ഏതാണ് କଣ का अस्ति हो
    ਦੀ ਦੇ ਦਾ ਹੈ છે এর এরের හෝ
    """.split()
)
_GENITIVE_SUFFIX = frozenset(
    {
        "ची", "चा", "चे", "च्या", "की", "का", "के", "को",
        "ाची", "ाचा", "ाचे", "ाच्या", "ের", "ৰ", "દી", "ની",
        "வின்", "ின்", "രുടെ", "स्य",
    }
)
_SUBJECT_STRIP = (
    "வின்", "ின்", "രുടെ", "स्य", "ाची", "ার", "ের", "ది", "ની", "ਦੀ", "को",
)
_CAPITAL_MARK = re.compile(
    r"राजधानी|தலைநகரம்|રાજધાની|ਰਾਜਧਾਨੀ|دارالحکومت|"
    r"തലസ്ഥാനം|ರಾಜಧಾನಿ|ରାଜଧାନୀ|রাজধানী|ৰাজধানী"
)
_DEFN_MARK = re.compile(
    r"\(noun\)|\bthe noun\b|\bhas \d+ senses?\b|"
    r"\b(?:defined as|consists of|consisting of|refers to)\b",
    re.I,
)


def query_subjects(query: str) -> list[str]:
    """Place/entity the question is about, minus 'capital' / question words."""
    return [
        t
        for t in content_tokens(query)
        if t not in _CAPITAL_LEX
        and t not in _SUBJECT_STOP
        and t not in STOP
        and len(t) >= 3
    ]


def _subject_stems(subject: str) -> list[str]:
    sl = (subject or "").lower()
    out = [sl]
    for suf in _SUBJECT_STRIP:
        if sl.endswith(suf) and len(sl) - len(suf) >= 3:
            out.append(sl[: -len(suf)])
            break
    return out


def mentions_subject(text: str, subject: str) -> bool:
    """True if `subject` appears as itself or a genitive (भारत/भारताची), not भारतीय."""
    if not subject or not text:
        return False
    stems = _subject_stems(subject)
    for t in content_tokens(text):
        for sl in stems:
            if t == sl:
                return True
            if len(t) > len(sl) and t.startswith(sl) and t[len(sl) :] in _GENITIVE_SUFFIX:
                return True
    return False


def capital_alignment(query: str, text: str) -> float:
    """+1 if the *owner* of 'capital' is the queried place.

    Only the last few tokens before राजधानी / first tokens after 'capital of'
    count. A 36-char window falsely treated 'India's Tamil Nadu' as India.
    """
    subjects = query_subjects(query)
    if not subjects or not text:
        return 0.0
    owners: list[str] = []
    for m in _CAPITAL_MARK.finditer(text):
        owners.append(" ".join(content_tokens(text[: m.start()])[-3:]))
    for m in re.finditer(r"capital of\s+", text, flags=re.I):
        owners.append(" ".join(content_tokens(text[m.end() : m.end() + 48])[:4]))
    if not owners:
        return 0.0
    if any(any(mentions_subject(w, s) for s in subjects) for w in owners):
        return 1.0
    return -1.0


def definition_alignment(text: str, query: str = "") -> float:
    """Definitional lead-in vs number-heavy factoid.

    `is a` only counts when it defines the query's own headword
    ("a fugue is a piece…"), not "Otis is an American company".
    """
    head = (text or "")[:360]
    if _DEFN_MARK.search(head):
        return 0.85
    if query:
        for tok in content_tokens(query):
            if len(tok) < 4:
                continue
            stem = tok[:-1] if tok.endswith("s") and tok.isascii() else tok
            if re.search(
                rf"\b{re.escape(stem)}\w{{0,4}}\s+(?:is|are|means)\s+an?\b",
                head,
                flags=re.I,
            ):
                return 0.85
    if len(re.findall(r"\d", head)) >= 4:
        return -0.55
    return 0.0


def looks_like_question(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    if s.endswith("?") or s.endswith("؟"):
        return True
    first = s.split("\n", 1)[0].strip()
    return first.endswith("?") or first.endswith("؟")


def bm25_query_tokens(query: str) -> list[str]:
    """Tokenize a query for BM25, with a cheap English plural fold."""
    toks = tokenize(query)
    extra: list[str] = []
    for t in toks:
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss") and t.isascii():
            extra.append(t[:-1])
    return toks + extra


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
        r"\b(what is|what are|what does|define|meaning of|meaning|describe)\b"
        r"|क्या है|क्या होता|என்ன|શું છે|ਕੀ ਹੈ|کیا ہے|എന്താണ്|ಯಾವುದು|କଣ|কী\?| কি\?",
        q,
    ):
        return "DESCRIPTION"
    if re.search(r"\b(which|who|what company|what organization)\b", q):
        return "ENTITY"
    return "UNKNOWN"
