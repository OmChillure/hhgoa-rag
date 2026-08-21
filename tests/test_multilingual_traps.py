"""Cross-language trap cases: definition vs headline / subtype / tangent / metaphor.

These are the failure modes that showed up in English first (Cabinet Net Worth,
Nurun, shoujo-ai, GOFAI, culinary capital) and must stay fixed in every
MSMARCO-XI language the reader claims to support.
"""

from voice_rag.generation.extractive import extract
from voice_rag.textutil import identity_person_query, infer_query_type
from voice_rag.types import Chunk, ChunkStrategy, Hit


def _hit(text: str, rank: int, cid: str) -> Hit:
    ch = Chunk(
        chunk_id=cid,
        parent_id=cid,
        strategy=ChunkStrategy.PASSAGE,
        text=text,
        language="xx",
    )
    return Hit(chunk=ch, score=0.4, rank=rank, origin="bm25", parent_text=text)


def _answer(query: str, gold: str, bait: str, qtype: str | None = None) -> str:
    qt = qtype or infer_query_type(query)
    ans, _, _ = extract(query, [_hit(bait, 0, "b"), _hit(gold, 1, "g")], qt)
    return ans


def test_who_is_not_height_or_net_worth_multilingual():
    rows = [
        (
            "डोनाल्ड ट्रंप कौन है?",
            "डोनाल्ड ट्रंप एक अमेरिकी राजनीतिज्ञ और रियल एस्टेट डेवलपर हैं।",
            "डोनाल्ड ट्रंप कैबिनेट नेट वर्थ। ऊंचाई 6 फुट 2 इंच।",
            ["राजनीतिज्ञ", "डेवलपर"],
            ["कैबिनेट", "ऊंचाई"],
        ),
        (
            "डोनाल्ड ट्रम्प कोण आहे?",
            "डोनाल्ड ट्रम्प हे एक अमेरिकन राजकारणी आहेत.",
            "डोनाल्ड ट्रम्प यांची उंची 6 फुट आहे.",
            ["राजकारणी"],
            ["उंची"],
        ),
        (
            "ਡੋਨਲਡ ਟਰੰਪ ਕੌਣ ਹੈ?",
            "ਡੋਨਲਡ ਟਰੰਪ ਇੱਕ ਅਮਰੀਕੀ ਰਾਜਨੇਤਾ ਹਨ।",
            "ਡੋਨਲਡ ਟਰੰਪ ਕੈਬਿਨੇਟ ਨੈੱਟ ਵਰਥ। ਉਚਾਈ 6 ਫੁੱਟ।",
            ["ਰਾਜਨੇਤਾ"],
            ["ਕੈਬਿਨੇਟ", "ਉਚਾਈ"],
        ),
        (
            "ডোনাল্ড ট্রাম্প কে?",
            "ডোনাল্ড ট্রাম্প একজন মার্কিন রাজনীতিবিদ।",
            "ডোনাল্ড ট্রাম্পের উচ্চতা ৬ ফুট ২ ইঞ্চি।",
            ["রাজনীতিবিদ"],
            ["উচ্চতা"],
        ),
        (
            "டொனால்ட் டிரம்ப் யார்?",
            "டொனால்ட் டிரம்ப் ஒரு அமெரிக்க அரசியல்வாதி ஆவார்.",
            "டொனால்ட் டிரம்பின் உயரம் 6 அடி 2 அங்குலம்.",
            ["அரசியல்வாதி"],
            ["உயரம்"],
        ),
        (
            "ڈونلڈ ٹرمپ کون ہے؟",
            "ڈونلڈ ٹرمپ ایک امریکی سیاستدان ہیں۔",
            "ڈونلڈ ٹرمپ کی قد 6 فٹ ہے۔ کابینہ مالیت۔",
            ["سیاستدان"],
            ["قد", "کابینہ"],
        ),
    ]
    for q, gold, bait, must, forbid in rows:
        assert identity_person_query(q), q
        ans = _answer(q, gold, bait)
        for m in must:
            assert m in ans, (q, ans)
        for f in forbid:
            assert f not in ans, (q, ans)


def test_what_is_not_website_designer_multilingual():
    rows = [
        (
            "स्पेसएक्स क्या है?",
            "स्पेसएक्स एक अमेरिकी एयरोस्पेस कंपनी है जो रॉकेट बनाती है।",
            "स्पेसएक्स की वेबसाइट नूरुन ने बनाई थी।",
            ["कंपनी", "रॉकेट"],
            ["नूरुन", "वेबसाइट"],
        ),
        (
            "ਸਪੇਸਐਕਸ ਕੀ ਹੈ?",
            "ਸਪੇਸਐਕਸ ਇੱਕ ਅਮਰੀਕੀ ਏਰੋਸਪੇਸ ਕੰਪਨੀ ਹੈ ਜੋ ਰਾਕੇਟ ਬਣਾਉਂਦੀ ਹੈ।",
            "ਸਪੇਸਐਕਸ ਦੀ ਵੈੱਬਸਾਈਟ ਨੂਰੁਨ ਨੇ ਡਿਜ਼ਾਈਨ ਕੀਤੀ।",
            ["ਕੰਪਨੀ", "ਰਾਕੇਟ"],
            ["ਨੂਰੁਨ", "ਵੈੱਬਸਾਈਟ"],
        ),
        (
            "স্পেসএক্স কী?",
            "স্পেসএক্স একটি আমেরিকান মহাকাশ সংস্থা যা রকেট তৈরি করে।",
            "স্পেসএক্সের ওয়েবসাইট নুরুন ডিজাইন করেছে।",
            ["সংস্থা", "রকেট"],
            ["নুরুন", "ওয়েবসাইট"],
        ),
        (
            "ஸ்பேஸ்எக்ஸ் என்றால் என்ன?",
            "ஸ்பேஸ்எக்ஸ் ஒரு அமெரிக்க விண்வெளி நிறுவனம் ஆகும்.",
            "ஸ்பேஸ்எக்ஸ் இணையதளத்தை நூரன் வடிவமைத்தார்.",
            ["நிறுவனம்"],
            ["நூரன்", "இணையதள"],
        ),
        (
            "اسپیس ایکس کیا ہے؟",
            "اسپیس ایکس ایک امریکی خلائی کمپنی ہے جو راکٹ بناتی ہے۔",
            "اسپیس ایکس کی ویب سائٹ نورون نے ڈیزائن کی۔",
            ["کمپنی", "راکٹ"],
            ["نورون", "ویب"],
        ),
    ]
    for q, gold, bait, must, forbid in rows:
        assert infer_query_type(q) == "DESCRIPTION", q
        ans = _answer(q, gold, bait)
        for m in must:
            assert m in ans, (q, ans)
        for f in forbid:
            assert f not in ans, (q, ans)


def test_what_is_not_subtype_approach_multilingual():
    rows = [
        (
            "what is artificial intelligence?",
            "Artificial intelligence (AI) is an area of computer science.",
            "GOFAI is an approach to achieving artificial intelligence.",
            ["computer"],
            ["gofai", "approach"],
        ),
        (
            "कृत्रिम बुद्धिमत्ता क्या है?",
            "कृत्रिम बुद्धिमत्ता कंप्यूटर विज्ञान की एक शाखा है जो बुद्धिमान मशीनें बनाती है।",
            "गोफाई कृत्रिम बुद्धिमत्ता प्राप्त करने का एक दृष्टिकोण है।",
            ["कंप्यूटर", "विज्ञान"],
            ["गोफाई", "दृष्टिकोण"],
        ),
        (
            "কৃত্রিম বুদ্ধিমত্তা কী?",
            "কৃত্রিম বুদ্ধিমত্তা কম্পিউটার বিজ্ঞানের একটি শাখা।",
            "গোফাই কৃত্রিম বুদ্ধিমত্তা অর্জনের একটি পদ্ধতি।",
            ["কম্পিউটার", "বিজ্ঞান"],
            ["গোফাই", "পদ্ধতি"],
        ),
    ]
    for q, gold, bait, must, forbid in rows:
        ans = _answer(q, gold, bait).lower()
        for m in must:
            assert m.lower() in ans, (q, ans)
        for f in forbid:
            assert f.lower() not in ans, (q, ans)


def test_capital_not_metaphor_multilingual():
    rows = [
        (
            "भारत की राजधानी क्या है?",
            "नई दिल्ली भारत की राजधानी है।",
            "मुंबई भारत की वित्तीय राजधानी कही जाती है।",
            ["दिल्ली"],
            ["मुंबई"],
        ),
        (
            "ਭਾਰਤ ਦੀ ਰਾਜਧਾਨੀ ਕੀ ਹੈ?",
            "ਨਵੀਂ ਦਿੱਲੀ ਭਾਰਤ ਦੀ ਰਾਜਧਾਨੀ ਹੈ।",
            "ਮੁੰਬਈ ਭਾਰਤ ਦੀ ਵਿੱਤੀ ਰਾਜਧਾਨੀ ਕਹੀ ਜਾਂਦੀ ਹੈ।",
            ["ਦਿੱਲੀ"],
            ["ਮੁੰਬਈ"],
        ),
        (
            "ভারতের রাজধানী কী?",
            "নয়াদিল্লি ভারতের রাজধানী।",
            "মুম্বাইকে ভারতের আর্থিক রাজধানী বলা হয়।",
            ["দিল্লি"],
            ["মুম্বাই"],
        ),
        (
            "இந்தியாவின் தலைநகரம் எது?",
            "புது தில்லி இந்தியாவின் தலைநகரம் ஆகும்.",
            "மும்பை இந்தியாவின் நிதித் தலைநகரம் என்று அழைக்கப்படுகிறது.",
            ["தில்லி"],
            ["மும்பை"],
        ),
        (
            "فرانس کا دارالحکومت کیا ہے؟",
            "پیرس فرانس کا دارالحکومت ہے۔",
            "لیون فرانس کا پاک دارالحکومت کہا جاتا ہے۔",
            ["پیرس"],
            ["لیون"],
        ),
    ]
    for q, gold, bait, must, forbid in rows:
        assert infer_query_type(q) == "LOCATION", q
        ans = _answer(q, gold, bait)
        for m in must:
            assert m in ans, (q, ans)
        for f in forbid:
            assert f not in ans, (q, ans)


def test_where_is_classified_across_scripts():
    assert infer_query_type("गोवा कहाँ है?") == "LOCATION"
    assert infer_query_type("गोवा कुठे आहे?") == "LOCATION"
    assert infer_query_type("ਗੋਆ ਕਿੱਥੇ ਹੈ?") == "LOCATION"
    assert infer_query_type("গোয়া কোথায়?") == "LOCATION"
    gold = "गोवा भारत के पश्चिमी तट पर स्थित एक राज्य है।"
    bait = "खंडवा मुख्य रेल लाइन पर है, गोवा के लिए दैनिक ट्रेनें हैं।"
    ans = _answer("गोवा कहाँ है?", gold, bait)
    assert "राज्य" in ans or "तट" in ans
    assert "खंडवा" not in ans


def test_how_tall_hindi_not_bio():
    assert infer_query_type("ओबामा की ऊंचाई कितनी है?") == "NUMERIC"
    ans = _answer(
        "ओबामा की ऊंचाई कितनी है?",
        "बराक ओबामा की ऊंचाई 6 फुट 1 इंच (185 सेमी) है।",
        "बराक ओबामा एक अमेरिकी राजनीतिज्ञ हैं जो 44वें राष्ट्रपति थे।",
        "NUMERIC",
    )
    assert "6 फुट" in ans or "185" in ans
    assert "राजनीतिज्ञ" not in ans
