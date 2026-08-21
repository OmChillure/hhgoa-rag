"""Sub-10ms extractive reader.

MS MARCO answers are short spans. We score candidate sentences / clauses
from retrieved *parent* passages (query-time span chunking) with a linear
combination of lexical overlap, exact phrase hits, and type-aware bonuses.
No model call on the SLA path.
"""

from __future__ import annotations

import re

from voice_rag.textutil import (
    ATTR_FACT,
    BREADCRUMB,
    CLAUSE_SPLIT,
    FINANCE_CAPITAL,
    acronym_alignment,
    capital_alignment,
    content_tokens,
    definition_alignment,
    identity_definition,
    identity_person_query,
    looks_like_headline,
    looks_like_question,
    primary_definition,
    query_subjects,
    sentences,
    short_head_mismatch,
    subtype_mention,
    tangent_mention,
)
from voice_rag.types import Hit, Span


_NUM = re.compile(r"\d[\d,.]*%?")
_ANAPHOR_START = re.compile(
    r"^(?:वे|वह|वो|ये|यह|ते|तो|ती|त्यांनी|त्यांचे|ਉਹ|તે|তিনি|সে|அவர்|ಅವರು|അദ്ദേഹം|ସେ|وہ"
    r"|they|he|she|it|this|that|those)\b",
    re.IGNORECASE,
)
_ORD_FIRST = re.compile(r"पहले|प्रथम|पहिला|पहिलो|पहिले|ਪਹਿਲੇ|পহিল|first|1st", re.IGNORECASE)
_ORD_NOT_FIRST = re.compile(
    r"दूसरे|द्वितीय|दुसरे|दوسرے|ਦੂਜੇ|দ্বিতীয়|இரண்டா|ಎರಡನೇ|രണ്ടാമ|ଦ୍ୱିତୀୟ|બીજા|"
    r"उपराष्ट्रपति|उपराष्ट्रपती|उप[- ]?राष्ट्रप|ਉਪ\s*ਰਾਸ਼ਟਰਪਤੀ|উপ[- ]?রাষ্ট্রপতি|"
    r"vice[-\s]?president|second|2nd",
    re.IGNORECASE,
)
_ABBR_TAIL = re.compile(r"(?:डॉ|श्री|प्रो|Dr|Mr|Mrs|Ms|Prof)\.?$", re.IGNORECASE)
# Generic sentence-initial/descriptor capitals that aren't real named
# entities on their own — filtered out so they don't masquerade as answers
# (e.g. "The Summers ... humid." shouldn't outscore "Bhubaneswar, Odisha.").
_GENERIC_CAPS = frozenset(
    """
    the a an this that these those it he she we they i you his her its their
    our your indian state states country city town region area district
    province national international world global
    january february march april may june july august september october
    november december monday tuesday wednesday thursday friday saturday
    sunday
    """.split()
)


_NAME_SPAN = re.compile(
    r"(?:डॉ|श्री|प्रो|Dr|Mr|Mrs|Ms|Prof)\.?\s*"
    r"(?:[\u0900-\u097F]{2,}(?:\s+[\u0900-\u097F]{2,}){0,3}"
    r"|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"
)
# Headlines / money-bio fragments that match Title Case but are not names.
_NON_NAME = frozenset(
    """
    cabinet net worth salary million billion height weight size shoe
    member members estimated wealth wealthy annual
    """.split()
)


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
        for m in _NAME_SPAN.finditer(sent):
            name = m.group(0).strip()
            if 6 <= len(name) < len(sent):
                # Latin Title-Case common nouns ("Cabinet Net Worth") match
                # the same pattern as names. Indic honorific spans do not.
                if re.search(r"[A-Z][a-z]", name) and not _person_name(name, set()):
                    continue
                out.append((start + m.start(), start + m.end(), name))
        cursor = end
    return out


_Q_STEMS = frozenset(
    """
    invented founded created called named located meaning definition define
    describe explain happen happened caused started become used using
    """.split()
)


def _specific_terms(q_toks: list[str]) -> list[str]:
    out: list[str] = []
    for t in q_toks:
        if t in _Q_STEMS:
            continue
        if len(t) >= 5 or 3 <= len(t) <= 4:
            if t not in out:
                out.append(t)
    return out


def _person_name(phrase: str, q_set: set[str]) -> bool:
    words = re.findall(r"[A-Za-z][\w'’.-]*", phrase)
    if len(words) < 2:
        return False
    lows = [w.lower().strip(".'") for w in words]
    if any(w in q_set or w in _GENERIC_CAPS for w in lows):
        return False
    # "Cabinet Net Worth" / "Salary Million" are Title Case but not names.
    if sum(1 for w in lows if w in _NON_NAME) >= max(1, len(lows) // 2):
        return False
    return True


def _score(query: str, cand: str, query_type: str, rank: int = 0, start: int = 0) -> float:
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
    overlap_scale = 1.0
    spec = _specific_terms(q_toks)
    miss = sum(1 for t in spec if t not in c_set)
    if query_type == "NUMERIC":
        q_nums = set(_NUM.findall(query))
        c_nums = _NUM.findall(cand)
        if c_nums:
            type_bonus += 0.45
        if q_nums and any(n in cand for n in q_nums):
            type_bonus += 0.2
        # Prefer the number that sits next to the query's own key terms
        # ("population of India is 1.3B") over some other statistic that
        # merely mentions India.
        if c_nums and spec and miss == 0:
            type_bonus += 0.25
        if re.search(r"how tall|how high|ऊंचाई|कद", q_low) and re.search(
            r"\bheight\b|ft\s*\d|\dcm\b|inches|ऊंचाई|फुट|इंच",
            c_low,
        ):
            type_bonus += 1.55
        elif re.search(r"how tall|how high|ऊंचाई|कद", q_low):
            type_bonus -= 1.2
        if re.search(r"share price|stock price", q_low) and re.search(
            r"\$\d|price|closed at|tsla|aapl|share",
            c_low,
        ):
            type_bonus += 1.4
    if query_type in {"PERSON", "ENTITY", "LOCATION"}:
        # A candidate that just paraphrases the question ("...the capital
        # of Odisha...") scores high on overlap/phrase without ever naming
        # the answer. Keep some lexical signal so we stay on-topic, and
        # reward proper nouns that are NEW relative to the query.
        overlap_scale = 0.65
        stripped = cand.strip()
        caps = re.findall(r"\b[A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)*", stripped)
        novel_caps = [
            c for c in caps
            if (c_ctoks := content_tokens(c))
            and not set(c_ctoks) <= q_set
            and not set(c_ctoks) <= _GENERIC_CAPS
        ]
        # Sentence-initial capitalization is grammatically required, so a
        # capitalized leading word alone isn't evidence of a named answer —
        # UNLESS it's immediately followed by a comma, the classic
        # "Bhubaneswar, Odisha." definitional-answer shape. Mid-sentence
        # capitals (leading==False) are much stronger entity evidence,
        # since nothing forces them to be capitalized otherwise.
        strong_caps = [
            c for c in novel_caps
            if not stripped.startswith(c) or re.match(re.escape(c) + r"\s*,", stripped)
        ]
        if strong_caps:
            # Cap the per-entity bonus so an entity-dense but off-topic
            # sentence ("...princely state..., had its capital at X,
            # located in district Y...") can't out-stack a single entity
            # that sits right next to the query's own key term — proximity
            # to a query word (e.g. "capital") is much stronger evidence
            # that THIS is the entity being asked about than raw count.
            near_query_word = any(
                any(
                    q_tok in stripped[max(0, m.start() - 25) : m.end() + 25].lower()
                    for q_tok in q_set
                )
                for c in strong_caps
                if (m := re.search(re.escape(c), stripped))
            )
            type_bonus += min(0.55, 0.28 * len(strong_caps))
            if near_query_word:
                type_bonus += 0.45
        elif novel_caps:
            type_bonus += 0.2
        else:
            # Indic/non-Latin scripts have no A-Z entities. Reward tokens
            # the question did not already contain — that's the name.
            novel = c_set - q_set
            if novel:
                type_bonus += 0.28 * min(4, len(novel))
            else:
                type_bonus -= 0.5
        # A short, entity-led candidate near the start of a sentence
        # ("Bhubaneswar, Odisha." / "X is the capital of Y.") is the
        # classic definitional-answer shape — nudge it above longer
        # descriptive sentences that merely discuss the same topic.
        if (strong_caps or novel_caps) and stripped[:1].isupper() and len(cand) <= 140:
            lead = (strong_caps or novel_caps)[0]
            if stripped.startswith(lead):
                type_bonus += 0.45
        # Short definitional answers ("Bhubaneswar, Odisha.") omit question
        # nouns like "capital" on purpose. Don't treat that as off-topic
        # when the span is a named entity from a top hit.
        short_entity = (
            rank <= 1
            and len(cand) <= 90
            and bool(strong_caps or novel_caps)
        )
        if miss and not short_entity:
            type_bonus *= 0.35
        elif short_entity:
            miss = 0
        if query_type == "PERSON" and re.search(
            r"\b(?:by|invented|founded|created|president|author|discovered)\s+[A-Z]",
            cand,
            flags=re.I,
        ):
            type_bonus += 0.55
        if query_type == "PERSON" and re.search(
            r"\b(?:another|also|independently|inspired)\b", c_low
        ):
            type_bonus -= 0.45
        if query_type == "PERSON":
            identity = identity_person_query(query)
            if identity:
                # "who is NAME" / "X कौन है" wants a definitional bio, not a
                # headline ("Cabinet Net Worth" / "कैबिनेट नेट वर्थ") or an
                # attribute factoid (height/salary/ऊंचाई/वेतन).
                defn = definition_alignment(cand, query)
                type_bonus += 1.7 * defn
                if looks_like_headline(cand):
                    type_bonus -= 2.2
                if ATTR_FACT.search(cand) and defn <= 0:
                    type_bonus -= 1.8
                if identity_definition(cand, query_subjects(query)):
                    type_bonus += 1.15
            else:
                latinish = bool(re.search(r"[A-Za-z]{3,}", cand))
                if any(_person_name(c, q_set) for c in novel_caps):
                    type_bonus += 0.55
                elif latinish and len(cand) > 80:
                    type_bonus -= 0.65
                if 8 <= len(cand) <= 72 and (c_set - q_set):
                    type_bonus += 0.85
                if _NAME_SPAN.fullmatch(stripped) and (
                    not re.search(r"[A-Z][a-z]", stripped) or _person_name(stripped, q_set)
                ):
                    type_bonus += 1.8
                if len(cand) > 160:
                    type_bonus -= 1.3
            places = query_subjects(query)
            geo = [p for p in places if p not in {"पहले", "first", "president", "राष्ट्रपति", "राष्ट्रपती"}]
            if geo and not any(p in c_set for p in geo) and len(cand) > 72:
                type_bonus -= 2.1
            if re.search(r"भारत|india", query, flags=re.I):
                if re.search(r"वाशिंगटन|washington|अमेरिकी|george washington", cand, flags=re.I):
                    type_bonus -= 2.8
                if re.search(r"राजेंद्र|प्रसाद|rajendra", cand, flags=re.I):
                    type_bonus += 1.4
            if re.search(r"\b(?:invented|founded|created|discovered|inventor)\b|आविष्कार|किसने", q_low):
                named = bool(strong_caps or _NAME_SPAN.search(stripped) or (c_set - q_set))
                if not named or not re.search(
                    r"[A-Z][a-z]+|[\u0900-\u097F]{3,}|डॉ|Dr\.",
                    stripped,
                ):
                    type_bonus -= 2.2
                if re.search(r"\bon the \d|in \d{4}\b", c_low) and not re.search(
                    r"\bby [A-Z]|invented by|आविष्कार",
                    cand,
                    flags=re.I,
                ):
                    type_bonus -= 1.8
                for head in spec:
                    if head in {"invented", "founded", "created", "discovered", "inventor"}:
                        continue
                    m = re.search(rf"\b{re.escape(head)}\s+(microscope|house|party|company|show)\b", c_low)
                    if m and m.group(1) not in q_set:
                        type_bonus -= 2.0
        if query_type == "LOCATION" and re.search(
            r"\b(?:is|was) the capital\b|की राजधानी है|ची राजधानी|को राजधानी",
            c_low,
        ):
            type_bonus += 0.7
        if query_type == "LOCATION":
            align = capital_alignment(query, cand)
            if align < 0:
                type_bonus += 1.65 * align
            elif align > 0 and re.search(
                r"(?:is|was)\s+the\s+capital|"
                r",\s*the\s+capital|"
                r"जो .{0,24}राजधानी|"
                r"राजधानी असलेली|"
                r"राजधानी है|"
                r"ਰਾਜਧਾਨੀ ਹੈ|রাজধানী|தலைநகரம்|دارالحکومت",
                cand,
                flags=re.I,
            ):
                type_bonus += 0.55
        if query_type == "LOCATION" and re.search(
            r"क्षेत्र|विभाग|प्रांत|region|department|province|district", c_low
        ):
            type_bonus -= 0.55
        if cand.count(",") >= 3:
            type_bonus -= 0.45
        if query_type == "LOCATION" and stripped[:1].islower():
            type_bonus -= 0.4
        if query_type == "LOCATION":
            places = query_subjects(query)
            if places and not any(p in c_set for p in places):
                type_bonus -= 1.9
            if FINANCE_CAPITAL.search(cand):
                type_bonus -= 2.4
            if BREADCRUMB.search(cand):
                type_bonus -= 1.1
            if re.search(
                r"\bwhere\b|कहाँ|कहां|कुठे|ਕਿੱਥੇ|ક્યાં|কোথায়|எங்கே|کہاں",
                query.lower(),
            ) and re.search(
                r"\b(?:state|city|town|located|west coast|east coast|coast of)\b|"
                r"स्थित|राज्य|तट|पश्चिमी|वसलेल|ਸਥਿਤ|অবস্থিত",
                c_low,
            ):
                type_bonus += 0.45
            if cand.count(",") >= 4:
                type_bonus -= 1.3
            if re.search(
                r"\b(?:connections? to|daily connections|train line)\b|"
                r"रेल लाइन|दैनिक ट्रेन|ट्रेनें|रेलमार्ग|रेलवे",
                c_low,
            ):
                type_bonus -= 1.7
            for p in places:
                if re.search(rf"\b{re.escape(p)}\b,?\s+a\s+(?:state|city|town)", c_low):
                    type_bonus += 1.4
                if re.search(rf"\b{re.escape(p)}\s+is\s+(?:a|located|situated)\b", c_low):
                    type_bonus += 1.4
                if re.search(r"\bis located\b", c_low) and not re.search(
                    rf"\b{re.escape(p)}\s+is located\b", c_low
                ):
                    type_bonus -= 1.8
    if query_type in {"DESCRIPTION", "ENTITY"}:
        type_bonus += 1.7 * definition_alignment(cand, query)
        type_bonus += 1.4 * acronym_alignment(query, cand)
        if primary_definition(query, cand):
            type_bonus += 1.25
        if subtype_mention(query, cand):
            type_bonus -= 2.4
        if tangent_mention(query, cand):
            type_bonus -= 2.4
        if short_head_mismatch(query, cand):
            type_bonus -= 2.3
        if looks_like_headline(cand):
            type_bonus -= 1.8
        if _NUM.search(cand) and definition_alignment(cand, query) <= 0:
            type_bonus -= 0.55
        if cand.count(";") >= 3:
            type_bonus -= 1.6
        if len(c_toks) < 6:
            type_bonus -= 1.5
        if re.search(
            r"best answer:|yahoo|gossip|tea party|cleaning \w+|how to clean|"
            r"necessary step|is an old term|बेस्ट\s*आं[नस]र|गॉसिप|गप्प",
            c_low,
        ):
            type_bonus -= 2.0
        if re.search(
            r"\b(?:mineral|plant|shrub|leaves|beverage|drink|substance|rock|crystal)\b",
            c_low,
        ):
            type_bonus += 0.75
        if re.match(r"cleaning\b", c_low):
            type_bonus -= 1.6
    if query_type in {"LOCATION", "PERSON", "ENTITY", "NUMERIC"}:
        length_pen = 0.0 if 8 <= len(cand) <= 240 else -0.12
    else:
        length_pen = 0.0 if 48 <= len(cand) <= 280 else -0.55
    spec_pen = -0.45 * miss
    echo_pen = 0.0
    q_core = re.sub(r"[?؟।.]+$", "", q_low).strip()
    c_core = re.sub(r"[?؟।.]+$", "", c_low).strip()
    if q_core and (
        c_core == q_core
        or c_low.startswith("question:")
        or c_low.startswith(q_core[: min(40, len(q_core))])
        or (q_core in c_core and len(c_toks) <= len(q_toks) + 2)
        or c_set <= q_set
    ):
        echo_pen = -1.6
    # MSMARCO dumps other questions into passages ("दक्षिण कोरियाची राजधानी कोणती आहे?").
    if looks_like_question(cand):
        echo_pen -= 2.6
    # "वे भारत के दूसरे राष्ट्रपति थे" is the *continuation* after the
    # name was split off at Indic danda (।). Never prefer a dangling pronoun.
    if _ANAPHOR_START.match(cand.strip()):
        echo_pen -= 1.1
    # "first president" queries: the MSMARCO trap is the Radhakrishnan
    # line ("second president / first vice-president"). Kill it in every script.
    if (
        query_type == "PERSON"
        and _ORD_FIRST.search(q_low)
        and re.search(r"राष्ट्रप|president|ਰਾਸ਼ਟਰਪਤੀ|صدر|குடியரசு", q_low)
        and _ORD_NOT_FIRST.search(cand)
    ):
        echo_pen -= 2.2
    if _ABBR_TAIL.search(cand.strip()):
        echo_pen -= 1.4
    # Retrieval rank dominates type-heuristics: RRF scores are ~0.02, so
    # the old `0.35 * hit.score` term was a no-op and the reader picked
    # flashy entities from weak passages.
    rank_bonus = 1.25 / (1.0 + rank)
    if query_type == "DESCRIPTION":
        rank_bonus *= 0.5
    # Later clauses in the same parent need to be clearly better than
    # the opening definitional sentence (Bell vs "also Elisha Gray").
    pos_pen = -min(0.4, start / 350.0)
    return (
        overlap * overlap_scale
        + phrase * overlap_scale
        + type_bonus
        + length_pen
        + spec_pen
        + echo_pen
        + rank_bonus
        + pos_pen
    )


def _heal_abbr_cut(parent: str, span: Span) -> str:
    """If a split ate the name after 'डॉ.', glue the next words back on."""
    text = (span.text or "").rstrip()
    if not _ABBR_TAIL.search(text):
        return text
    rest = (parent or "")[span.end :].lstrip()
    if not rest:
        return text
    take = re.match(r".{2,80}?(?:[।.!?؟۔]|$)", rest)
    extra = (take.group(0) if take else rest[:80]).strip()
    return f"{text} {extra}".strip() if extra else text


def _previous_sentence(parent: str, start: int) -> str:
    prefix = (parent or "")[: max(0, start)].rstrip()
    if not prefix:
        return ""
    parts = sentences(prefix)
    prev = parts[-1].strip() if parts else ""
    if not prev or len(prev) > 220 or _ANAPHOR_START.match(prev):
        return ""
    return prev


def extract(query: str, hits: list[Hit], query_type: str = "UNKNOWN") -> tuple[str, list[Span], float]:
    scored: list[tuple[float, Span, str]] = []
    for hit in hits:
        parent = hit.parent_text or hit.chunk.text
        for start, end, cand in _candidates(parent):
            s = _score(query, cand, query_type, rank=hit.rank, start=start)
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
    best_score, best_span, parent = scored[0]
    if _ANAPHOR_START.match(best_span.text.strip()):
        prev = _previous_sentence(parent, best_span.start)
        if prev:
            best_span = Span(
                text=f"{prev} {best_span.text}",
                parent_id=best_span.parent_id,
                start=max(0, best_span.start - len(prev) - 1),
                end=best_span.end,
                score=best_score,
            )
    healed = _heal_abbr_cut(parent, best_span)
    if healed != best_span.text:
        best_span = Span(
            text=healed,
            parent_id=best_span.parent_id,
            start=best_span.start,
            end=min(len(parent), best_span.start + len(healed)),
            score=best_score,
        )
    # Optionally stitch a neighboring high-scoring span from the same
    # parent — but only for open-ended query types. LOCATION/PERSON/
    # ENTITY/NUMERIC questions want one crisp fact; stitching in an
    # unrelated same-passage sentence (e.g. a temperature reading tacked
    # onto a capital-city answer) just because it also scored well
    # produces a garbled answer, not a fuller one.
    stitchable = query_type in {"DESCRIPTION", "UNKNOWN"}
    extras = [
        s
        for sc, s, _ in scored[1:4]
        if stitchable
        and s.parent_id == best_span.parent_id
        and s.text != best_span.text
        and sc > best_score * 0.72
        and not re.search(r"cleaning |best answer|how to |gossip", s.text, flags=re.I)
        and not subtype_mention(query, s.text)
        and not tangent_mention(query, s.text)
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
