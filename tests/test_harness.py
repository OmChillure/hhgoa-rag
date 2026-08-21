from voice_rag.generation.extractive import extract
from voice_rag.textutil import infer_query_type
from voice_rag.types import Chunk, ChunkStrategy, Hit


def _hit(text: str, rank: int, cid: str = "c") -> Hit:
    ch = Chunk(
        chunk_id=cid,
        parent_id=cid,
        strategy=ChunkStrategy.PASSAGE,
        text=text,
        language="en",
    )
    return Hit(chunk=ch, score=0.4, rank=rank, origin="bm25", parent_text=text)


def test_extractive_picks_supported_span():
    parent = (
        "Paris is the capital and most populous city of France. "
        "It is located on the river Seine in the north of the country."
    )
    hits = [_hit(parent, 0)]
    answer, spans, conf = extract("what is the capital of france?", hits, "LOCATION")
    assert "Paris" in answer
    assert spans
    assert conf > 0


def test_extractive_prefers_top_hit_over_entity_bait():
    gold = "Bhubaneswar, Odisha. The Summers in the capital of the Indian State of Odisha are hot and humid."
    bait = (
        "Rajput Provinces of India - Bonai (Princely State) Bonai State was a princely "
        "state during the British Raj and had its capital at Bonaigarh, located in the "
        "present-day Sundergarh district of Odisha."
    )
    answer, _, _ = extract(
        "what is the capital of odisha?",
        [_hit(gold, 0, "g"), _hit(bait, 1, "b")],
        "LOCATION",
    )
    assert "Bhubaneswar" in answer
    assert "Bonaigarh" not in answer


def test_extractive_prefers_inventor_over_related_person():
    gold = "The first telephone was made by Alexander Graham Bell in 1876. Historians also mention Elisha Gray."
    bait = (
        "At the U.S. Centennial Exposition, Emile Berliner had seen a Bell Company "
        "telephone demonstrated and was inspired to find ways to improve the newly invented telephone."
    )
    answer, _, _ = extract(
        "who invented the telephone?",
        [_hit(gold, 0, "g"), _hit(bait, 2, "b")],
        "PERSON",
    )
    assert "Alexander Graham Bell" in answer
    assert "Emile Berliner" not in answer


def test_extractive_hindi_capital_prefers_new_delhi():
    chennai = (
        "चेन्नई भारतीय राज्य तमिलनाडु की राजधानी है। "
        "यह एक प्रमुख सांस्कृतिक केंद्र है और भारत का पाँचवाँ सबसे बड़ा शहर है।"
    )
    delhi = "नई दिल्ली, जो भारत की राजधानी है, दिल्ली में एक क्षेत्र है। नई दिल्ली भारत सरकार का केंद्र है।"
    answer, _, _ = extract(
        "भारत की राजधानी क्या है",
        [_hit(chennai, 0, "c"), _hit(delhi, 2, "d")],
        "LOCATION",
    )
    assert "दिल्ली" in answer
    assert "चेन्नई" not in answer


def test_extractive_marathi_skips_other_country_question():
    gold = (
        "१९९२ मध्ये राष्ट्रीय राजधानी क्षेत्र कायद्यांतर्गत "
        "भारताची राजधानी असलेली नवी दिल्ली हे एक राज्य बनले."
    )
    bait = (
        "दक्षिण कोरियाची राजधानी कोणती आहे? नकाशावर सोलचे स्थान. "
        "सोल हे दक्षिण कोरियाचे राजधानी शहर आहे."
    )
    answer, _, _ = extract(
        "भारताची राजधानी कोणती आहे",
        [_hit(gold, 0, "g"), _hit(bait, 1, "b")],
        "LOCATION",
    )
    assert "दिल्ली" in answer
    assert "कोरिया" not in answer


def test_extractive_what_is_tea_not_a_catalog():
    catalog = (
        "Berry Tea; Caramel Tea; Chamomile Tea; Citrus Tea; Cinnamon Tea; "
        "Earl Grey Tea; Floral Tea; Ginger Tea; Rose Tea; Vanilla Tea"
    )
    definition = (
        "TEA LEAF (noun). The noun TEA LEAF has 1 sense: 1. dried leaves of "
        "the tea shrub; used to make tea."
    )
    prose = (
        "Oolong tea is a traditional Chinese tea. It's made from the leaves of "
        "the Camellia sinensis plant, the same plant used to make green tea and black tea."
    )
    answer, _, _ = extract(
        "what is tea?",
        [_hit(catalog, 0, "c"), _hit(prose, 1, "p"), _hit(definition, 2, "d")],
        "DESCRIPTION",
    )
    low = answer.lower()
    assert "tea" in low
    assert len(answer.split()) >= 5
    assert "Berry Tea" not in answer
    assert "dried leaves" in low or "camellia" in low or "chinese tea" in low


def test_extractive_meaning_prefers_definition():
    gold = (
        "ELEVATOR (noun) The noun ELEVATOR has 2 senses: 1. lifting device "
        "consisting of a platform or cage that is raised and lowered mechanically "
        "in a vertical shaft."
    )
    bait = (
        "Burj Khalifa has a total of 57 elevators and eight escalators. "
        "Among them are the world's tallest service elevator, which has a capacity of 5.500 kg."
    )
    answer, _, _ = extract(
        "elevators meaning",
        [_hit(bait, 0, "b"), _hit(gold, 1, "g")],
        "DESCRIPTION",
    )
    assert "lifting device" in answer.lower()
    assert "Burj" not in answer


def test_who_invented_is_person():
    assert infer_query_type("elevators meaning") == "DESCRIPTION"
    assert infer_query_type("who invented the telephone?") == "PERSON"
    assert infer_query_type("फ्रांस की राजधानी क्या है?") == "LOCATION"
    assert infer_query_type("भारत के पहले राष्ट्रपति कौन थे?") == "PERSON"
    assert infer_query_type("பிரான்சின் தலைநகரம் என்ன?") == "LOCATION"
    assert infer_query_type("فرانس کا دارالحکومت کیا ہے؟") == "LOCATION"


def test_extractive_does_not_cut_name_before_anaphor():
    cut = (
        "डॉ. सर्वपल्ली राधाकृष्णन (1888-1975)। "
        "वे भारत के दूसरे राष्ट्रपति और भारत के पहले उपराष्ट्रपति थे।"
    )
    gold = (
        "प्रथम गणतंत्र दिवस समारोहों के झलक और भारत के पहले राष्ट्रपति के अविस्मरणीय "
        "दृश्यों ने भारतीय इतिहास में एक नए युग की शुरुआत को चिह्नित किया जब भारत "
        "गणराज्य का जन्म डॉ. राजेंद्र प्रसाद के पहले राष्ट्रपति के रूप में शपथ के साथ हुआ था।"
    )
    answer, _, _ = extract(
        "भारत के पहले राष्ट्रपति कौन थे?",
        [_hit(cut, 0, "c"), _hit(gold, 0, "g")],
        "PERSON",
    )
    assert not answer.startswith("वे")
    assert "राजेंद्र" in answer
    assert not answer.rstrip().endswith("डॉ.")
    assert "उपराष्ट्रपति" not in answer


def test_extractive_punjabi_skips_vice_president_trap():
    trap = "ਉਹ ਭਾਰਤ ਦੇ ਦੂਜੇ ਰਾਸ਼ਟਰਪਤੀ ਅਤੇ ਭਾਰਤ ਦੇ ਪਹਿਲੇ ਉਪ ਰਾਸ਼ਟਰਪਤੀ ਸਨ।"
    gold = "ਡਾ. ਰਾਜੇਂਦਰ ਪ੍ਰਸਾਦ ਭਾਰਤ ਦੇ ਪਹਿਲੇ ਰਾਸ਼ਟਰਪਤੀ ਸਨ ਅਤੇ 1950 ਤੋਂ 1962 ਤੱਕ ਰਹੇ।"
    answer, _, _ = extract(
        "ਭਾਰਤ ਦੇ ਪਹਿਲੇ ਰਾਸ਼ਟਰਪਤੀ ਕੌਣ ਸਨ?",
        [_hit(trap, 0, "t"), _hit(gold, 0, "g")],
        "PERSON",
    )
    assert "ਰਾਜੇਂਦਰ" in answer or "ਪ੍ਰਸਾਦ" in answer
    assert "ਉਪ" not in answer


def test_sentences_keep_doctor_title():
    from voice_rag.textutil import sentences

    parts = sentences("जब भारत गणराज्य का जन्म डॉ. राजेंद्र प्रसाद के पहले राष्ट्रपति के रूप में शपथ के साथ हुआ था।")
    assert any("राजेंद्र" in p and "डॉ." in p for p in parts)


def test_extractive_hindi_does_not_echo_question():
    parent = (
        "फ्रांस की राजधानी क्या है? "
        "पेरिस फ्रांस की राजधानी और देश का सबसे बड़ा शहर है।"
    )
    answer, _, _ = extract(
        "फ्रांस की राजधानी क्या है?",
        [_hit(parent, 0)],
        "LOCATION",
    )
    assert "पेरिस" in answer
    assert answer.strip(" ?।") != "फ्रांस की राजधानी क्या है"


def test_extractive_prefers_is_the_capital_sentence():
    gloss = (
        "Orly - a southeastern suburb of Paris; site of an international airport serving Paris. "
        "capital of France, City of Light, French capital, Paris - the capital and largest city of France; "
        "and international center of culture and commerce."
    )
    gold = (
        "Paris, France. The Paris office serves as the Company's European Headquarters. "
        "Paris is the capital of France and the country's largest city."
    )
    answer, _, _ = extract(
        "what is the capital of france?",
        [_hit(gloss, 0, "g"), _hit(gold, 1, "p")],
        "LOCATION",
    )
    assert "Paris" in answer
    assert "City of Light" not in answer


def test_extractive_goa_not_finance_capital():
    finance = (
        "Factors Affecting the Cost of Capital. The marginal cost of capital "
        "(MCC) is the cost of the last dollar of capital raised. WACC = (wd)(kd)(1-t)."
    )
    geo = (
        "Asia > South Asia > India > Western India > Goa. Goa, a state on "
        "India's West coast, is a former Portuguese colony with a rich history."
    )
    answer, _, _ = extract(
        "capital of goa",
        [_hit(finance, 0, "f"), _hit(geo, 1, "g")],
        "LOCATION",
    )
    assert "Goa" in answer
    assert "WACC" not in answer
    assert "cost of capital" not in answer.lower()


def test_extractive_where_is_goa_prefers_state_sentence():
    parent = (
        "Asia > South Asia > India > Western India > Goa. Goa, a state on "
        "India's West coast, is a former Portuguese colony with a rich history. "
        "Spread over 3,700 square kilometres with a population of approximately "
        "1.4 million, Goa is small by Indian standards."
    )
    answer, _, _ = extract("where is goa", [_hit(parent, 0)], "LOCATION")
    assert "state" in answer.lower()
    assert "West" in answer or "west" in answer.lower()
    assert answer.count(">") < 2


def test_extractive_names_the_president():
    title = "Glimpses of the First Republic Day Celebrations and India's First President."
    gold = (
        "Rajendra Prasad (1884-1963) was the first President the Republic of India. "
        "He held this post from 26 January 1950 to 13th May 1963."
    )
    answer, _, _ = extract(
        "who was the first president of india?",
        [_hit(title, 0, "t"), _hit(gold, 1, "g")],
        "PERSON",
    )
    assert "Rajendra Prasad" in answer
