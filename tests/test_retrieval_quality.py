from voice_rag.retrieval.hybrid import has_all_specific, lexical_relevance, rrf
from voice_rag.textutil import capital_alignment, detect_language, infer_query_type


def test_lexical_what_is_prefers_definition_not_tangent_or_homonym():
    sx = "what is spacex?"
    web = (
        "The SpaceX website was designed by a San Fransisco design company Nurun "
        "(Nurun - Design, Human Centered Thinking and Digital Products)."
    )
    rockets = "SpaceX designs, manufactures and launches rockets and spacecraft."
    assert lexical_relevance(sx, rockets) > lexical_relevance(sx, web)

    ai = "what is an ai?"
    manga = "So, shoujo ai is Girl's Love/GL Girls who love girls."
    cs = "Artificial intelligence (AI) is an area of computer science."
    assert lexical_relevance(ai, cs) > lexical_relevance(ai, manga)

    aiq = "what is artificial intelligence?"
    gofai = (
        "In artificial intelligence research, GOFAI (Good Old-Fashioned "
        "Artificial Intelligence) is an approach to achieving artificial intelligence."
    )
    area = "Artificial intelligence (AI) is an area of computer science."
    assert lexical_relevance(aiq, area) > lexical_relevance(aiq, gofai)


def test_rrf_boosts_shared_ids():
    fused = rrf([["a", "b", "c"], ["c", "a", "d"]])
    assert fused["a"] > fused["b"]
    assert fused["c"] > fused["b"]
    assert fused["a"] > fused["d"]


def test_lexical_prefers_phrase_and_penalizes_question_echo():
    query = "what is the population of india?"
    echo = "Question: What is the population of Chennai? Answer: Chennai, India (Administrative unit: Tamil Nadu)."
    gold = "The current population of India is 1,338,868,927 as of Sunday, April 9, 2017."
    bait = "What is the cost Ancestry DNA test in India The cost varies depending on how many population compared."
    assert lexical_relevance(query, gold) > lexical_relevance(query, echo)
    assert lexical_relevance(query, gold) > lexical_relevance(query, bait)


def test_detect_language_disambiguates_shared_scripts():
    assert detect_language("what is the capital of india") == "en"
    assert detect_language("भारत की राजधानी क्या है") == "hi"
    assert detect_language("भारताची राजधानी कोणती आहे") == "mr"
    assert detect_language("भारतको राजधानी के हो") == "ne"
    assert detect_language("भारतस्य राजधानी का अस्ति") == "sa"
    assert detect_language("ভারতের রাজধানী কোথায়") == "bn"
    assert detect_language("ভাৰতৰ ৰাজধানী কি") == "as"
    assert detect_language("இந்தியாவின் தலைநகரம் எது") == "ta"


def test_capital_alignment_binds_subject():
    q = "भारत की राजधानी क्या है"
    assert capital_alignment(q, "नई दिल्ली, जो भारत की राजधानी है") > 0
    assert capital_alignment(q, "चेन्नई भारतीय राज्य तमिलनाडु की राजधानी है") < 0
    assert capital_alignment("भारताची राजधानी कोणती आहे", "भारताची राजधानी असलेली नवी दिल्ली") > 0
    assert capital_alignment("भारताची राजधानी कोणती आहे", "दक्षिण कोरियाची राजधानी कोणती आहे?") < 0
    assert capital_alignment("what is the capital of india", "New Delhi, the capital of India") > 0
    assert capital_alignment(
        "what is the capital of india",
        "Chennai is the capital of the Indian state of Tamil Nadu",
    ) < 0
    assert capital_alignment(
        "ਭਾਰਤ ਦੀ ਰਾਜਧਾਨੀ ਕੀ ਹੈ",
        "ਚੇਨਈ ਭਾਰਤ ਦੇ ਤਾਮਿਲਨਾਡੂ ਰਾਜ ਦੀ ਰਾਜਧਾਨੀ ਹੈ।",
    ) < 0
    assert capital_alignment(
        "ਭਾਰਤ ਦੀ ਰਾਜਧਾਨੀ ਕੀ ਹੈ",
        "ਨਵੀਂ ਦਿੱਲੀ, ਜੋ ਕਿ ਭਾਰਤ ਦੀ ਰਾਜਧਾਨੀ ਹੈ, ਦਿੱਲੀ ਦਾ ਇੱਕ ਖੇਤਰ ਹੈ।",
    ) > 0


def test_infer_meaning_is_description():
    assert infer_query_type("elevators meaning") == "DESCRIPTION"


def test_lexical_requires_specific_term():
    query = "what direction does phloem flow"
    miss = "Direction of flow of current is opposite to the direction of flow of electrons."
    hit = "Phloem transports sugars; the direction of phloem flow is from source to sink."
    assert lexical_relevance(query, hit) > lexical_relevance(query, miss)
    assert has_all_specific(query, hit)
    assert not has_all_specific(query, miss)
