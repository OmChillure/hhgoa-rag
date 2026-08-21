from voice_rag.generation.extractive import extract
from voice_rag.textutil import identity_person_query, infer_query_type
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


def test_extractive_telephone_names_bell_not_just_the_date():
    date = "The telephone was invented on the 10th March 1876."
    gold = "The telephone was invented by Alexander Graham Bell, a Scottish inventor."
    answer, _, _ = extract(
        "who invented telephone",
        [_hit(date, 0, "d"), _hit(gold, 1, "g")],
        "PERSON",
    )
    assert "Bell" in answer
    assert "March" not in answer


def test_extractive_invented_light_not_microscope():
    scope = (
        "Who invented the light microscope? The first light microscope was invented "
        "by Dutch spectacle makers Hans Jansen and his son Zacharias in the late 16th century."
    )
    bulb = (
        "Thomas Edison Lightbulb Let there be Light! Historians agree that Thomas Edison "
        "was not the inventor of the electric light bulb, but he did produce the first "
        "commercially viable one."
    )
    answer, _, _ = extract(
        "who invented light",
        [_hit(scope, 0, "s"), _hit(bulb, 1, "b")],
        "PERSON",
    )
    assert "microscope" not in answer.lower()
    assert "Edison" in answer or "Thomas" in answer


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


def test_extractive_quartz_is_mineral_not_cleaning():
    parent = (
        "Quartz is a mineral that is used in a variety of manners, including "
        "the making of jewelry and glass. When quartz is mined, it contains dirt. "
        "Cleaning quartz is a necessary step along the way of making it into a finished product."
    )
    answer, _, _ = extract("what is quartz", [_hit(parent, 0)], "DESCRIPTION")
    assert "mineral" in answer.lower()
    assert not answer.lower().startswith("cleaning")


def test_extractive_tea_not_gossip():
    gossip = "Best Answer: tea is an old term for gossip, which came from the idea of what goes on at a tea party."
    real = "TEA LEAF (noun). dried leaves of the tea shrub; used to make tea."
    answer, _, _ = extract(
        "what is tea",
        [_hit(gossip, 0, "g"), _hit(real, 1, "r")],
        "DESCRIPTION",
    )
    assert "gossip" not in answer.lower()
    assert "leaf" in answer.lower() or "shrub" in answer.lower() or "leaves" in answer.lower()


def test_extractive_india_president_not_washington():
    gold = (
        "प्रथम गणतंत्र दिवस समारोहों के झलक और भारत के पहले राष्ट्रपति के अविस्मरणीय "
        "दृश्यों ने भारतीय इतिहास में एक नए युग की शुरुआत को चिह्नित किया जब भारत "
        "गणराज्य का जन्म डॉ. राजेंद्र प्रसाद के पहले राष्ट्रपति के रूप में शपथ के साथ हुआ था।"
    )
    bait = (
        "अमेरिकी राष्ट्रपति सेना के प्रथम प्रधान सेनापति थे? जॉर्ज वाशिंगटन पहले थे, "
        "क्योंकि वह पहले राष्ट्रपति थे।"
    )
    answer, _, _ = extract(
        "भारत के पहले राष्ट्रपति कौन थे?",
        [_hit(gold, 0, "g"), _hit(bait, 1, "b")],
        "PERSON",
    )
    assert "राजेंद्र" in answer or "प्रसाद" in answer
    assert "वाशिंगटन" not in answer


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
    assert infer_query_type("tell me about tea") == "DESCRIPTION"
    assert infer_query_type("capital goa") == "LOCATION"
    assert infer_query_type("tea") == "DESCRIPTION"
    assert infer_query_type("डोनाल्ड ट्रंप कौन है?") == "PERSON"
    assert infer_query_type("डोनाल्ड ट्रम्प कोण आहे?") == "PERSON"
    assert infer_query_type("ਡੋਨਲਡ ਟਰੰਪ ਕੌਣ ਹੈ?") == "PERSON"
    assert infer_query_type("ডোনাল্ড ট্রাম্প কে?") == "PERSON"
    assert infer_query_type("டொனால்ட் டிரம்ப் யார்?") == "PERSON"
    assert infer_query_type("ڈونلڈ ٹرمپ کون ہے؟") == "PERSON"
    assert infer_query_type("donald trump kaun hai") == "PERSON"
    assert infer_query_type("how tall is barack obama?") == "NUMERIC"
    assert infer_query_type("what is tesla share price?") == "NUMERIC"
    assert identity_person_query("डोनाल्ड ट्रंप कौन है?")
    assert identity_person_query("ਡੋਨਲਡ ਟਰੰਪ ਕੌਣ ਹੈ?")
    assert identity_person_query("ডোনাল্ড ট্রাম্প কে?")
    assert identity_person_query("டொனால்ட் டிரம்ப் யார்?")
    assert identity_person_query("ڈونلڈ ٹرمپ کون ہے؟")
    assert identity_person_query("डोनाल्ड ट्रम्प कोण आहे?")
    assert not identity_person_query("भारत के पहले राष्ट्रपति कौन थे?")
    assert not identity_person_query("टेलीफोन का आविष्कार किसने किया?")


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
    assert "गणतंत्र दिवस" not in answer


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


def test_extractive_where_is_goa_not_khandwa_train_list():
    train = (
        "Khandwa is located on the Main Train Line, with daily connections to "
        "Mumbai, Pune, Delhi, Goa, Cochin, Kolkata, Indore, Bhopal, Patna, "
        "Allahabad, Lucknow, Jammu, Hyderabad, and Bangalore."
    )
    geo = (
        "Asia > South Asia > India > Western India > Goa. Goa, a state on "
        "India's West coast, is a former Portuguese colony with a rich history."
    )
    answer, _, _ = extract(
        "where is goa located",
        [_hit(train, 0, "t"), _hit(geo, 1, "g")],
        "LOCATION",
    )
    assert "Khandwa" not in answer
    assert "Goa" in answer
    assert "state" in answer.lower() or "West" in answer or "west" in answer.lower()


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


def test_extractive_who_is_not_a_headline_or_height():
    salary = (
        "Donald Trump's Salary $60 Million Donald Trump net worth: Donald Trump is an "
        "American politician, real-estate developer, author and television personality "
        "who has a net worth of $3.1 billion. His annual salary for Celebrity Apprentice "
        "is $60 million."
    )
    height = (
        "How tall is Donald Trump? Donald Trump's height is 6ft 2in (188 cm). "
        "How much does Donald Trump weigh? Donald Trump weighs 198 lbs (90 kg)."
    )
    cabinet = (
        "Donald Trump's Cabinet Net Worth: How Much Each Member is Worth. "
        "Donald Trump's cabinet has an estimated net worth of $14 billion."
    )
    answer, _, _ = extract(
        "who is donald trump?",
        [_hit(salary, 0, "s"), _hit(height, 1, "h"), _hit(cabinet, 2, "c")],
        "PERSON",
    )
    low = answer.lower()
    assert "politician" in low or "developer" in low or "personality" in low
    assert "cabinet" not in low
    assert "height" not in low
    assert "6ft" not in low
    assert "shoe" not in low


def test_extractive_who_is_obama_not_net_worth_title():
    bait = "Obama Net Worth Barack Obama has a net worth of $40 million."
    gold = (
        "Barack Obama is an American politician who served as the 44th president "
        "of the United States from 2009 to 2017."
    )
    answer, _, _ = extract(
        "who is barack obama?",
        [_hit(bait, 0, "b"), _hit(gold, 1, "g")],
        "PERSON",
    )
    assert "politician" in answer.lower() or "president" in answer.lower()
    assert "Net Worth" not in answer
    assert "$40" not in answer


def test_extractive_hindi_who_is_not_cabinet_or_height():
    salary = (
        "डोनाल्ड ट्रंप का वेतन 60 मिलियन डॉलर डोनाल्ड ट्रंप नेट वर्थ: "
        "डोनाल्ड ट्रंप एक अमेरिकी राजनीतिज्ञ, रियल एस्टेट डेवलपर, लेखक और "
        "टेलीविजन व्यक्तित्व हैं जिनकी कुल संपत्ति 3.1 अरब डॉलर है।"
    )
    height = (
        "डोनाल्ड ट्रंप की ऊंचाई कितनी है? डोनाल्ड ट्रंप की ऊंचाई 6 फुट 2 इंच "
        "(188 सेमी) है। डोनाल्ड ट्रंप का वजन 90 किलो है।"
    )
    cabinet = (
        "डोनाल्ड ट्रंप कैबिनेट नेट वर्थ: प्रत्येक सदस्य कितना अमीर है। "
        "डोनाल्ड ट्रंप की कैबिनेट की कुल संपत्ति 14 अरब डॉलर आंकी गई है।"
    )
    answer, _, _ = extract(
        "डोनाल्ड ट्रंप कौन है?",
        [_hit(salary, 0, "s"), _hit(height, 1, "h"), _hit(cabinet, 2, "c")],
        "PERSON",
    )
    assert "राजनीतिज्ञ" in answer or "डेवलपर" in answer or "व्यक्तित्व" in answer
    assert "कैबिनेट" not in answer
    assert "ऊंचाई" not in answer
    assert "6 फुट" not in answer


def test_extractive_marathi_who_is_not_height():
    gold = "डोनाल्ड ट्रम्प हे एक अमेरिकन राजकारणी, रिअल-इस्टेट डेव्हलपर आणि दूरदर्शन व्यक्तिमत्त्व आहेत."
    bait = "डोनाल्ड ट्रम्प यांची उंची 6 फुट 2 इंच आहे. त्यांचे वजन 90 किलो आहे."
    answer, _, _ = extract(
        "डोनाल्ड ट्रम्प कोण आहे?",
        [_hit(bait, 0, "b"), _hit(gold, 1, "g")],
        "PERSON",
    )
    assert "राजकारणी" in answer or "डेव्हलपर" in answer or "व्यक्तिमत्त्व" in answer
    assert "उंची" not in answer


def test_extractive_what_is_spacex_not_website_designer():
    web = (
        "The SpaceX website was designed by a San Fransisco design company Nurun "
        "(Nurun - Design, Human Centered Thinking and Digital Products). While SpaceX "
        "hasn't been added to their client list yet, see tweet for details."
    )
    gold = (
        "SpaceX designs, manufactures and launches rockets and spacecraft. "
        "SpaceX was founded in 2002."
    )
    news = (
        "SpaceX Falcon 9 Rocket Flies Safely. The SpaceX Falcon 9 rocket roared "
        "successfully into the sky from Cape Canaveral this morning."
    )
    answer, _, _ = extract(
        "what is spacex?",
        [_hit(web, 0, "w"), _hit(gold, 1, "g"), _hit(news, 2, "n")],
        "DESCRIPTION",
    )
    low = answer.lower()
    assert "rocket" in low or "spacecraft" in low or "manufactur" in low
    assert "nurun" not in low
    assert "website" not in low
    assert "human centered" not in low


def test_extractive_what_is_ai_not_shoujo():
    manga = (
        "Ai = Love. Josei = Woman. So, shoujo ai is Girl's Love/GL Girls who love girls. "
        "Shounen ai is Boy's Love/BL Boy who love boys. People tend to use shounen ai and "
        "shoujo ai for stories that are not sexually explicit."
    )
    gold = (
        "Artificial Intelligence (AI) Artificial intelligence (AI) is an area of computer "
        "science that emphasizes the creation of intelligent machines that work and react "
        "like humans."
    )
    water = (
        "The Institute of Medicine determined that an adequate intake (AI) for men is "
        "roughly about 13 cups (3 liters) of total beverages a day."
    )
    answer, _, _ = extract(
        "what is an ai?",
        [_hit(manga, 0, "m"), _hit(gold, 1, "g"), _hit(water, 2, "w")],
        "DESCRIPTION",
    )
    low = answer.lower()
    assert "computer" in low or "intelligent" in low or "machines" in low
    assert "shoujo" not in low
    assert "girl's love" not in low
    assert "shounen" not in low
    assert "cups" not in low


def test_extractive_what_is_ai_not_gofai_approach():
    gofai = (
        "GOFAI. In artificial intelligence research, GOFAI (Good Old-Fashioned "
        "Artificial Intelligence) is an approach to achieving artificial intelligence. "
        "In the robotics research, the term is extended as GOFAIR."
    )
    gold = (
        "Artificial Intelligence (AI) Artificial intelligence (AI) is an area of computer "
        "science that emphasizes the creation of intelligent machines that work and react "
        "like humans."
    )
    answer, _, _ = extract(
        "what is artificial intelligence?",
        [_hit(gofai, 0, "gofai"), _hit(gold, 1, "gold")],
        "DESCRIPTION",
    )
    low = answer.lower()
    assert "computer science" in low or "intelligent machines" in low
    assert "gofai" not in low
    assert "approach" not in low


def test_extractive_what_is_tesla_not_client_list():
    web = (
        "The Tesla website was designed by a Brooklyn studio called Instrument. "
        "Tesla hasn't been added to their client list yet, see tweet for details."
    )
    gold = (
        "Tesla, Inc. is an American electric vehicle and clean energy company "
        "that designs and manufactures electric cars."
    )
    answer, _, _ = extract(
        "what is tesla?",
        [_hit(web, 0, "w"), _hit(gold, 1, "g")],
        "DESCRIPTION",
    )
    low = answer.lower()
    assert "electric" in low or "vehicle" in low or "company" in low
    assert "client list" not in low
    assert "tweet" not in low
    assert "instrument" not in low


def test_extractive_what_is_wifi_not_password():
    bait = "WiFi password for starbucks. How to hack wifi. Best answer: the password is on the receipt."
    gold = (
        "Wi-Fi is a family of wireless network protocols based on the IEEE 802.11 "
        "standards, used for local area networking of devices."
    )
    answer, _, _ = extract(
        "what is wifi?",
        [_hit(bait, 0, "b"), _hit(gold, 1, "g")],
        "DESCRIPTION",
    )
    low = answer.lower()
    assert "wireless" in low or "network" in low or "802.11" in low
    assert "password" not in low
    assert "hack" not in low


def test_extractive_what_is_instagram_not_howto():
    bait = "How to delete Instagram: go to settings and tap delete. Instagram coupon codes."
    gold = "Instagram is a photo and video sharing social networking service owned by Meta Platforms."
    answer, _, _ = extract(
        "what is instagram?",
        [_hit(bait, 0, "b"), _hit(gold, 1, "g")],
        "DESCRIPTION",
    )
    low = answer.lower()
    assert "photo" in low or "social" in low or "video" in low
    assert "delete" not in low
    assert "coupon" not in low


def test_how_tall_is_numeric_and_picks_height():
    assert infer_query_type("how tall is barack obama?") == "NUMERIC"
    bio = "Barack Obama is an American politician who served as the 44th president of the United States."
    height = "Barack Obama's height is 6ft 1in (185 cm)."
    answer, _, _ = extract(
        "how tall is barack obama?",
        [_hit(bio, 0, "b"), _hit(height, 1, "h")],
        "NUMERIC",
    )
    assert "6ft" in answer or "185" in answer or "height" in answer.lower()
    assert "politician" not in answer.lower()


def test_share_price_is_numeric():
    assert infer_query_type("what is tesla share price?") == "NUMERIC"
    company = "Tesla, Inc. is an American electric vehicle and clean energy company that designs cars."
    price = "Tesla share price today is $248.32. TSLA closed at $248 on the Nasdaq."
    answer, _, _ = extract(
        "what is tesla share price?",
        [_hit(company, 0, "c"), _hit(price, 1, "p")],
        "NUMERIC",
    )
    assert "248" in answer


def test_extractive_france_capital_not_culinary():
    gold = "पेरिस फ्रांस की राजधानी और देश का सबसे बड़ा शहर है।"
    bait = "ल्योन फ्रांस का एक प्रमुख शहर है। ल्योन फ्रांस की पाक राजधानी कही जाती है।"
    answer, _, _ = extract(
        "फ्रांस की राजधानी क्या है?",
        [_hit(bait, 0, "b"), _hit(gold, 1, "g")],
        "LOCATION",
    )
    assert "पेरिस" in answer
    assert "ल्योन" not in answer


def test_extractive_hindi_what_is_not_tangent():
    gold = "स्पेसएक्स एक अमेरिकी एयरोस्पेस कंपनी है जो रॉकेट और अंतरिक्ष यान बनाती है।"
    bait = "स्पेसएक्स की वेबसाइट नूरुन नामक डिज़ाइन कंपनी ने बनाई थी।"
    answer, _, _ = extract(
        "स्पेसएक्स क्या है?",
        [_hit(bait, 0, "b"), _hit(gold, 1, "g")],
        "DESCRIPTION",
    )
    assert "कंपनी" in answer or "रॉकेट" in answer or "अंतरिक्ष" in answer
    assert "नूरुन" not in answer
    assert "वेबसाइट" not in answer


def test_extractive_punjabi_who_is_not_net_worth_title():
    gold = "ਡੋਨਲਡ ਟਰੰਪ ਇੱਕ ਅਮਰੀਕੀ ਰਾਜਨੇਤਾ, ਰੀਅਲ-ਇਸਟੇਟ ਡਿਵੈਲਪਰ ਅਤੇ ਟੈਲੀਵਿਜ਼ਨ ਸ਼ਖਸੀਅਤ ਹਨ।"
    bait = "ਡੋਨਲਡ ਟਰੰਪ ਕੈਬਿਨੇਟ ਨੈੱਟ ਵਰਥ: ਹਰ ਮੈਂਬਰ ਦੀ ਕੁੱਲ ਜਾਇਦਾਦ।"
    answer, _, _ = extract(
        "ਡੋਨਲਡ ਟਰੰਪ ਕੌਣ ਹੈ?",
        [_hit(bait, 0, "b"), _hit(gold, 1, "g")],
        "PERSON",
    )
    assert "ਰਾਜਨੇਤਾ" in answer or "ਡਿਵੈਲਪਰ" in answer or "ਸ਼ਖਸੀਅਤ" in answer
    assert "ਕੈਬਿਨੇਟ" not in answer
