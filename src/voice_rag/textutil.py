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
    r"\s+(?:and then|but|however|although|because|which|where|while|;|—|–)\s+"
    r"|:\s+",
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


_TOKEN_STRIP = "।.!?؟۔:ः,;\"'()[]"


def tokenize(text: str) -> list[str]:
    # Danda / Arabic full stop live inside the Indic/Arabic Unicode
    # blocks, so WORD_RE would otherwise keep "हैं।" as a different
    # token from "हैं".
    return [
        cleaned
        for t in WORD_RE.findall(text or "")
        if (cleaned := t.lower().strip(_TOKEN_STRIP))
    ]


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
    कोणती आहे च्या ची चा चे क्याहे कोण
    क्या है எது என்ன கை કઈ કે ਕੀ کیا
    কোথায় কী কি কে ಯಾವುದು ഏതാണ് କଣ का अस्ति हो
    ਦੀ ਦੇ ਦਾ ਹੈ છે এর এরের හෝ
    कौन हैं कौन है कौन था कौन थे कोण आहे
    ਕੌਣ કોણ யார் ಯಾರು ആര് ആരാണ് କିଏ కెవరు ఎవరు کون
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
FINANCE_CAPITAL = re.compile(
    r"\bcost of capital\b|\bwacc\b|\bmarginal cost of capital\b|"
    r"\bweighted average\b|\bcost of (?:debt|equity)\b|"
    r"\blast dollar of capital\b",
    re.I,
)
BREADCRUMB = re.compile(r"(?:[A-Za-z][\w.'’ -]+\s*>\s*){2,}")

# "who is / who was" across MSMARCO-XI languages. Identity questions
# ("who is NAME") share this; inventor/president questions also match
# but are filtered out by ROLE_SEEK.
WHO_IS = re.compile(
    r"\bwho\s+(?:is|was|are|were)\b|"
    r"\b(?:kaun|kon)\s+(?:hai|hain|tha|the)\b|"
    r"कौन\s*(?:हैं|है|था|थे|थी)|कौन|"
    r"कोण\s*(?:आहेत|आहे|होते|होता)|कोण|"
    r"को\s*(?:हुन्|हो|थिए)|"
    r"ਕੌਣ\s*(?:ਹਨ|ਹੈ|ਸਨ|ਸੀ)|ਕੌਣ|"
    r"કોણ\s*(?:છે|હતા|હતો)|કોણ|"
    r"কে\s*(?:ছিলেন|হয়|হলেন)|কোন\s*আছিল|(?:^|[\s])কে(?:\s*[?।]|$)|"
    r"யார்|"
    r"ಯಾರು|"
    r"ആരാണ്|ആര്|"
    r"କିଏ|"
    r"ఎవరు|కెవరు|"
    r"کون\s*(?:ہیں|ہے|تھا|تھے)|کون",
    re.I,
)
ROLE_SEEK = re.compile(
    r"\b(?:invented|founded|created|discovered|inventor|president|minister|"
    r"prime|ceo|king|queen|captain|director|author|writer|leader|"
    r"first|1st|second|2nd)\b|"
    r"किसने|आविष्कार|आविष्कर्ता|"
    r"राष्ट्रपति|राष्ट्रपती|राष्ट्रप|उपराष्ट्र|"
    r"पहले|प्रथम|पहिला|पहिलो|पहिले|"
    r"प्रधानमंत्री|मंत्री|राजा|रानी|"
    r"ਰਾਸ਼ਟਰਪਤੀ|ਪਹਿਲੇ|ਮੰਤਰੀ|"
    r"রাষ্ট্রপতি|প্রথম|"
    r"குடியரசு|முதல்|"
    r"ರಾಷ್ಟ್ರಪತಿ|"
    r"രാഷ്ട്രപതി|"
    r"ରାଷ୍ଟ୍ରପତି|"
    r"صدر|پہلے|اختراع|"
    r"અધ્યક્ષ|પ્રથમ",
    re.I,
)
INDEF = re.compile(
    r"(?:^|[\s,])(?:एक|एखाद|ਇੱਕ|એક|একজন|এজন|এক|ఒక|ಒಬ್ಬ|ഒരു|ଜଣେ|ஒரு|ایک)\s",
)
_ZERO_COPULA_DEF = re.compile(r"একজন|এজন")
COPULA = re.compile(
    r"(?:हैं|है|था|थे|थी|होते|होता|आहेत|आहे|अस्ति|"
    r"ਹਨ|ਹੈ|ਸਨ|ਸੀ|છે|હતા|હતો|"
    r"হন|হয়|ছিলেন|আছেন|ছিল|"
    r"ஆவார்|ஆவர்|என்பவர்|"
    r"ಆಗಿದ್ದಾರೆ|ಆಗಿದ್ದಾನೆ|"
    r"ആണ്|ആയിരുന്നു|"
    r"ଅଟନ୍ତି|ଅଟେ|"
    r"ہیں|ہے|تھا|تھے|"
    r"\bis\b|\bare\b|\bwas\b|\bwere\b)",
    re.I,
)
ATTR_FACT = re.compile(
    r"\b(?:net worth|salary|height|weight|shoe size|cabinet)\b|"
    r"how tall|how much (?:does|is|each)|"
    r"'s height|'s salary|'s net worth|ft \d|\dlbs\b|\$\d|"
    r"नेट\s*वर्थ|कुल संपत्ति|सम्पत्ति|संपत्ति|"
    r"वेतन|सैलरी|तनख्वाह|"
    r"ऊंचाई|उंचाई|लंबाई|कद|"
    r"वजन|भार|"
    r"जूते|साइज़|साइज|"
    r"कैबिनेट|"
    r"निव्वळ\s*मालमत्ता|उंची|पगार|"
    r"ਕੁੱਲ ਜਾਇਦਾਦ|ਉਚਾਈ|ਤਨਖਾਹ|ਕੈਬਿਨੇਟ|"
    r"ચોખ્ખી સંપત્તિ|ઊંચાઈ|પગાર|"
    r"নেট\s*ওয়ার্থ|সম্পত্তি|উচ্চতা|বেতন|ক্যাবিনেট|"
    r"நிகர\s*மதிப்பு|உயரம்|சம்பளம்|"
    r"مالیت|قد|تنخواہ|کابینہ|"
    r"ಎತ್ತರ|ಸಂಬಳ|"
    r"ഉയരം|ശമ്പളം|"
    r"ଉଚ୍ଚତା|ଦରମା",
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
        and (len(t) >= 3 or (len(t) == 2 and t.isascii() and t.isalpha()))
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
    if _METAPHOR_CAPITAL.search(text) and not _METAPHOR_CAPITAL.search(query):
        return -1.0
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


_ORG_PRED = re.compile(
    r"\b(?:designs|manufactures|builds|provides|operates|makes|creates|"
    r"offers|develops|launches|produces|sells|runs)\b",
    re.I,
)
_TANGENT_NOUN = (
    r"website|web site|webpage|homepage|logo|twitter|account|app|"
    r"stock|share|tweet|client list|fan page|"
    r"वेबसाइट|वेबसाईट|जालस्थल|लोगो|"
    r"ਵੈੱਬਸਾਈਟ|ওয়েবসাইট|இணையதளம்|ویب\s*سائٹ"
)
_GEN = r"की|का|के|ची|चा|चे|ਦੀ|ના|এর|யின்"
_TANGENT_FREE = re.compile(
    r"\b(?:client list|see tweet|hasn't been added to their|"
    r"how to (?:delete|hack|install|download|remove|buy)|"
    r"(?:wifi |wi-?fi )?password|coupon codes?|whois|"
    r"buy this domain|lyrics:)\b|"
    r"कैसे\s*(?:डिलीट|हैक|हटा|डाउनलोड)|पासवर्ड|"
    r"बेस्ट\s*आं[नस]र|गॉसिप|गप्प|"
    r"কিভাবে|পাসওয়ার্ড|"
    r"ਕਿਵੇਂ|ਪਾਸਵਰਡ|"
    r"எப்படி|"
    r"کیسے|پاس\s*ورڈ",
    re.I,
)
_SUBTYPE = re.compile(
    r"\bis an?\s+(?:approach|method|technique|algorithm|subset|branch|"
    r"subfield|variant|style|framework|paradigm|extension)\s+"
    r"(?:to|of|for)\b|"
    r"प्राप्त करने का एक\s*(?:दृष्टिकोण|विधि|पद्धति|तरीका)|"
    r"(?:एक\s+)?(?:दृष्टिकोण|उपागम)\s*(?:है|हैं)|"
    r"অর্জনের একটি\s*(?:পদ্ধতি|পন্থা)|"
    r"அணுகுமுறை|"
    r"طریقہ\s*کار|"
    r"दृष्टिकोन",
    re.I,
)
_METAPHOR_CAPITAL = re.compile(
    r"(?:पाक|फैशन|सांस्कृतिक|खेल|फिल्म|वित्तीय|आर्थिक|व्यावसायिक)\s*राजधानी|"
    r"ਵਿੱਤੀ\s*ਰਾਜਧਾਨੀ|ਸੱਭਿਆਚਾਰਕ\s*ਰਾਜਧਾਨੀ|"
    r"আর্থিক\s*রাজধানী|সাংস্কৃতিক\s*রাজধানী|"
    r"நிதித்\s*தலைநகரம்|பண்பாட்டு\s*தலைநகரம்|"
    r"(?:مالی|ثقافتی|پاک)\s*دارالحکومت|"
    r"(?:culinary|fashion|cultural|film|sports|food|pak|financial)\s+capital",
    re.I,
)
def _query_heads(query: str) -> list[str]:
    """Content heads of a what/who question, including 2-letter acronyms (ai)."""
    out: list[str] = []
    for t in content_tokens(query):
        if t in _SUBJECT_STOP or t in STOP or t in _CAPITAL_LEX:
            continue
        if len(t) >= 3 or (len(t) == 2 and t.isascii() and t.isalpha()):
            if t not in out:
                out.append(t)
    return out or query_subjects(query)


def _short_acronym_head(query: str) -> str | None:
    """Single 2–3 letter head ('ai', 'gpu'). Not 'flow' in a multi-word question."""
    heads = _query_heads(query)
    if len(heads) != 1:
        return None
    h = heads[0]
    if 2 <= len(h) <= 3 and h.isascii() and h.isalpha():
        return h
    return None


def acronym_alignment(query: str, text: str) -> float:
    """+1 when a short query head is expanded in the passage: 'Artificial Intelligence (AI)'."""
    if not query or not text:
        return 0.0
    h = _short_acronym_head(query)
    if not h:
        return 0.0
    if re.search(rf"\({re.escape(h)}\)", text, flags=re.I):
        return 0.9
    if re.search(rf"\b{re.escape(h)}\s+\([^)]{{6,}}\)", text, flags=re.I):
        return 0.9
    return 0.0


def tangent_mention(query: str, text: str) -> bool:
    """True when the subject is only a modifier (SpaceX website designed by …)."""
    if not query or not text:
        return False
    for s in _query_heads(query):
        if re.search(
            rf"{re.escape(s)}\w{{0,4}}(?:'s)?(?:\s+(?:{_GEN}))?\s+(?:{_TANGENT_NOUN})",
            text,
            flags=re.I,
        ):
            return True
        if re.search(
            rf"\b{re.escape(s)}\w{{0,4}}\s+was\s+designed\s+by\b",
            text,
            flags=re.I,
        ):
            return True
    if _TANGENT_FREE.search(text):
        heads = _query_heads(query)
        tset = set(content_tokens(text))
        if heads and any(h in tset or h in text.lower() for h in heads):
            return True
    return False


def primary_definition(query: str, text: str) -> bool:
    """True when the copula's subject *is* the queried thing.

    'Artificial intelligence (AI) is an area of…' is primary.
    'GOFAI is an approach to achieving artificial intelligence' is not.
    """
    heads = _query_heads(query)
    if not heads or not text:
        return False
    if re.search(
        r"best answer:|is an old term|yahoo|gossip|tea party|how to |"
        r"बेस्ट\s*आं[नस]र|गॉसिप",
        text,
        flags=re.I,
    ):
        return False
    name = " ".join(heads)
    if re.search(
        rf"(?:^|[\s.])(?:the\s+)?{re.escape(name)}\s*(?:\([^)]{{1,40}}\)\s*)?"
        r"(?:is|are|was|were|means)\s+an?\b",
        text,
        flags=re.I,
    ):
        return True
    if 2 <= len(heads) <= 5 and all(h.isascii() and h.isalpha() for h in heads):
        acr = "".join(h[0] for h in heads)
        if re.search(
            rf"\({re.escape(acr)}\)\s*(?:is|are|was|were|means)\s+an?\b",
            text,
            flags=re.I,
        ):
            return True
    # Indic: the queried name must lead the sentence, then indef + copula.
    # "कृत्रिम बुद्धिमत्ता … एक शाखा है" yes; "गोफाई कृत्रिम बुद्धिमत्ता …" no.
    lead = content_tokens(text)
    if (
        lead
        and lead[0] == heads[0]
        and all(h in lead[: max(3, len(heads) + 1)] for h in heads[:2])
        and INDEF.search(text)
        and (COPULA.search(text) or _ZERO_COPULA_DEF.search(text))
    ):
        return True
    return False


def subtype_mention(query: str, text: str) -> bool:
    """True when the span defines a *kind of* the queried thing, not the thing."""
    if not query or not text or not _SUBTYPE.search(text):
        return False
    return not primary_definition(query, text)


def short_head_mismatch(query: str, text: str) -> bool:
    """True when a short head is used as another word (shoujo ai) not an acronym."""
    if not query or not text:
        return False
    h = _short_acronym_head(query)
    if not h:
        return False
    if re.search(rf"\({re.escape(h)}\)", text, flags=re.I):
        return False
    if re.search(rf"\b[a-z]{{4,}}\s+{re.escape(h)}\b", text):
        return True
    if re.search(rf"^{re.escape(h)}\s*=", text, flags=re.I):
        return True
    return False


def definition_alignment(text: str, query: str = "") -> float:
    """Definitional lead-in vs number-heavy factoid.

    `is a` only counts when it defines the query's own headword
    ("a fugue is a piece…"), not "Otis is an American company".
    """
    head = (text or "")[:360]
    if query and (
        tangent_mention(query, head)
        or short_head_mismatch(query, head)
        or subtype_mention(query, head)
    ):
        if len(re.findall(r"\d", head)) >= 4:
            return -0.55
        return 0.0
    if _DEFN_MARK.search(head):
        return 0.85
    if query:
        for tok in _query_heads(query):
            stem = tok[:-1] if tok.endswith("s") and tok.isascii() and len(tok) > 3 else tok
            if re.search(
                r"best answer:|yahoo|gossip|tea party|cleaning \w+|how to clean|"
                r"necessary step|is an old term",
                head,
                flags=re.I,
            ):
                continue
            if re.search(
                rf"\b{re.escape(stem)}\w{{0,4}}\s+(?:is|are|means)\s+an?\s+"
                r"(?:necessary|old|good|important|great|common|popular)\b",
                head,
                flags=re.I,
            ):
                continue
            if re.search(
                rf"\b{re.escape(stem)}\w{{0,4}}\s+(?:is|are|means)\s+an?\b",
                head,
                flags=re.I,
            ):
                return 0.85
            # "Artificial intelligence (AI) is an area…"
            if re.search(
                rf"\({re.escape(tok)}\)\s*(?:is|are|means)\s+an?\b",
                head,
                flags=re.I,
            ):
                return 0.85
            if re.search(
                rf"\b{re.escape(stem)}\w{{0,4}}\s+\([^)]{{0,40}}\)\s*(?:is|are|means)\s+an?\b",
                head,
                flags=re.I,
            ):
                return 0.85
            # "SpaceX designs, manufactures and launches rockets"
            if re.search(rf"\b{re.escape(stem)}\w{{0,4}}\s+", head, flags=re.I) and _ORG_PRED.search(head):
                if re.search(
                    rf"\b{re.escape(stem)}\w{{0,4}}\s+{_ORG_PRED.pattern}",
                    head,
                    flags=re.I,
                ):
                    return 0.85
            # Indic: "ट्रंप एक अमेरिकी राजनीतिज्ञ हैं" / "ট্রাম্প একজন রাজনীতিবিদ"
            if re.search(re.escape(stem), head, flags=re.I) and INDEF.search(head):
                if COPULA.search(head) or _ZERO_COPULA_DEF.search(head):
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


def identity_person_query(query: str) -> bool:
    """True when the person is already named and we need a description.

    "who is Donald Trump" / "ट्रंप कौन है" is identity.
    "who invented X" / "पहले राष्ट्रपति कौन थे" still want a short name.
    """
    if not WHO_IS.search(query or ""):
        return False
    return not ROLE_SEEK.search(query)


def looks_like_headline(text: str) -> bool:
    """Catalog/title fragment, not a sentence — Latin Title Case or Indic noun pile."""
    s = (text or "").strip().rstrip("।.!?؟۔:")
    if not s or len(s) > 80:
        return False
    if COPULA.search(s):
        return False
    words = [w for w in s.split() if w]
    if len(words) < 2:
        return False
    if re.search(r"[A-Za-z]", s):
        caps = sum(1 for w in words if w[:1].isupper() or w[:1].isdigit() or w[:1] in "$")
        if caps >= max(2, int(0.7 * len(words))):
            return True
    if ATTR_FACT.search(s) and len(words) <= 10:
        return True
    raw = (text or "").strip()
    if raw.endswith(":") or raw.endswith("ः"):
        return True
    return False


def identity_definition(cand: str, subjects: list[str]) -> bool:
    """Candidate defines the queried name: 'X is an…' / 'X एक … हैं'."""
    if not cand or not subjects:
        return False
    c_toks = content_tokens(cand)
    if not c_toks:
        return False
    if not any(s in c_toks for s in subjects[:2]):
        return False
    cl = cand.lower()
    name = " ".join(subjects[:2])
    if name and re.search(
        rf"^{re.escape(name)}\s+(?:is|was|are|were)\s+(?:an?\s+)?",
        cl,
    ):
        return True
    if INDEF.search(cand) and COPULA.search(cand):
        return True
    # Bengali/Assamese present-tense bios often drop the copula.
    return bool(_ZERO_COPULA_DEF.search(cand))


def infer_query_type(query: str) -> str:
    q = (query or "").lower()
    if re.search(
        r"\b(how many|how much|how tall|how high|how long|how heavy|how old|how wide|"
        r"what year|when was|when did|population|age|number of|percentage|percent|"
        r"share price|stock price)\b"
        r"|जनसंख्या|कितने|कितनी|कितना|ऊंचाई|সংখ্যা|எத்தனை|વસ્તી|ਆਬਾਦੀ|آبادی",
        q,
    ):
        return "NUMERIC"
    if re.search(
        r"\b(where|which city|which country|located|capital)\b"
        r"|राजधानी|தலைநகரம்|રાજધાની|ਰਾਜਧਾਨੀ|دارالحکومت|തലസ്ഥാനം|ರಾಜಧಾನಿ|ରାଜଧାନୀ|রাজধানী|ৰাজধানী"
        r"|कहाँ|कहां|कुठे|ਕਿੱਥੇ|ક્યાં|কোথায়|எங்கே|کہاں|ಎಲ್ಲಿ|എവിടെ",
        q,
    ):
        return "LOCATION"
    if re.search(
        r"\b(who is|who was|who were|who invented|which person|invented by|founded by|who)\b"
        r"|\b(?:kaun|kon)\s+(?:hai|hain|tha|the)\b"
        r"|किसने|कौन थे|कौन है|कौन था|कौन हैं|कौन|"
        r"कोण आहेत|कोण आहे|कोण होते|कोण|"
        r"को थिए|को हो|"
        r"யார்|કોણ|ਕੌਣ|کون |کون|"
        r"ആരാണ്|ആര്|ಯಾರು|କିଏ|"
        r"কে ছিলেন|কোন আছিল|(?:^|[\s])কে(?:\s*[?।]|$)|"
        r"ఎవరు|కెవరు",
        q,
    ):
        return "PERSON"
    if re.search(
        r"\b(what is|what are|what does|what|define|meaning of|meaning|describe|explain|tell me about|about)\b"
        r"|क्या है|क्या होता|क्या|म्हणजे काय|என்ன|શું છે|ਕੀ ਹੈ|کیا ہے|എന്താണ്|ಯಾವುದು|କଣ|কী\?| কি\?",
        q,
    ):
        return "DESCRIPTION"
    if re.search(r"\b(which|what company|what organization)\b", q):
        return "ENTITY"
    return "DESCRIPTION"
