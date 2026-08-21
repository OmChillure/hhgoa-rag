from voice_rag.generation.slm import SpanRewriter, rewrite_is_faithful
from voice_rag.harness.orchestrator import Harness
from voice_rag.types import Chunk, ChunkStrategy, Hit


def _hit(text: str, rank: int = 0, cid: str = "c") -> Hit:
    ch = Chunk(
        chunk_id=cid,
        parent_id=cid,
        strategy=ChunkStrategy.PASSAGE,
        text=text,
        language="en",
    )
    return Hit(chunk=ch, score=0.4, rank=rank, origin="bm25", parent_text=text)


class _FakeRetriever:
    def __init__(self, text: str) -> None:
        self.text = text

    def search(self, query: str, language: str = "en", query_type: str = "UNKNOWN"):
        return [_hit(self.text)]


class _FakeSLM:
    def __init__(self, text: str, ready: bool = True) -> None:
        self.ready = ready
        self.text = text
        self.calls = 0

    def rewrite(self, query: str, span: str, timeout_ms: float | None = None) -> str:
        self.calls += 1
        return self.text


def test_faithful_rewrite_rejects_hallucination_and_dangling():
    span = "Goa, a state on India's West coast, is a former Portuguese colony."
    assert rewrite_is_faithful("Goa is a state on India's West coast.", span)
    assert not rewrite_is_faithful("Goa is a country in the South East region of", span)
    assert not rewrite_is_faithful("the cost of capital", span)
    assert not rewrite_is_faithful("tea", "dried leaves of the tea shrub used to make tea")


def test_faithful_rewrite_rejects_headline_fragments():
    bio = (
        "Donald Trump is an American politician, real-estate developer, author "
        "and television personality who has a net worth of $3.1 billion."
    )
    assert not rewrite_is_faithful("Cabinet Net Worth", bio)
    assert not rewrite_is_faithful("Donald Trump net worth", bio)
    assert not rewrite_is_faithful("Donald Trump's Salary $60 Million", bio)
    assert rewrite_is_faithful(
        "Donald Trump is an American politician and real-estate developer.",
        bio,
    )
    hi = (
        "डोनाल्ड ट्रंप एक अमेरिकी राजनीतिज्ञ, रियल एस्टेट डेवलपर, लेखक "
        "और टेलीविजन व्यक्तित्व हैं जिनकी कुल संपत्ति 3.1 अरब डॉलर है।"
    )
    assert not rewrite_is_faithful("कैबिनेट नेट वर्थ", hi)
    assert not rewrite_is_faithful("डोनाल्ड ट्रंप नेट वर्थ", hi)
    assert rewrite_is_faithful("डोनाल्ड ट्रंप एक अमेरिकी राजनीतिज्ञ हैं।", hi)
    web = (
        "The SpaceX website was designed by a San Fransisco design company Nurun "
        "(Nurun - Design, Human Centered Thinking and Digital Products)."
    )
    assert not rewrite_is_faithful(
        "Nurun (Nurun - Design, Human Centered Thinking)",
        web,
        "what is spacex?",
    )
    assert rewrite_is_faithful(
        "SpaceX designs, manufactures and launches rockets and spacecraft.",
        "SpaceX designs, manufactures and launches rockets and spacecraft. SpaceX was founded in 2002.",
        "what is spacex?",
    )


def test_rewrite_is_noop_until_loaded():
    slm = SpanRewriter()
    assert slm.ready is False
    assert slm.rewrite("what is paris", "Paris is the capital of France.") == ""


def test_generate_mode_when_model_is_grounded():
    parent = "Paris is the capital and most populous city of France."
    slm = _FakeSLM("Paris is the capital of France.")
    result = Harness(_FakeRetriever(parent), slm=slm).run("what is the capital of france?")
    assert slm.calls == 1
    assert result.answer.mode == "generate"
    assert result.answer.text == "Paris is the capital of France."
    assert result.answer.grounded
    assert any(t.name == "generate_answer" for t in result.timings)


def test_extractive_fallback_when_model_hallucinates():
    parent = "Paris is the capital and most populous city of France."
    slm = _FakeSLM("The answer is definitely Berlin, Germany.")
    result = Harness(_FakeRetriever(parent), slm=slm).run("what is the capital of france?")
    assert slm.calls == 1
    assert result.answer.mode == "generate"
    assert "Paris" in result.answer.text
    assert "Berlin" not in result.answer.text


def test_extractive_when_slm_missing():
    parent = "Paris is the capital and most populous city of France."
    slm = _FakeSLM("", ready=False)
    result = Harness(_FakeRetriever(parent), slm=slm).run("what is the capital of france?")
    assert slm.calls == 0
    assert result.answer.mode == "generate"
    assert "Paris" in result.answer.text


class _LangRetriever:
    def __init__(self, by_lang: dict) -> None:
        self.by_lang = by_lang
        self.calls: list[str] = []

    def search(self, query: str, language: str = "en", query_type: str = "UNKNOWN"):
        self.calls.append(language)
        return list(self.by_lang.get(language, []))


def test_empty_indic_retrieve_retries_english():
    gold = "पेरिस फ्रांस की राजधानी और देश का सबसे बड़ा शहर है।"
    retr = _LangRetriever({"hi": [], "en": [_hit(gold)]})
    result = Harness(retr, slm=_FakeSLM("", ready=False)).run("फ्रांस की राजधानी क्या है?")
    assert retr.calls == ["hi", "en"]
    assert "पेरिस" in result.answer.text


def test_weak_indic_hits_do_not_retry_english():
    retr = _LangRetriever(
        {"hi": [_hit("unrelated quartz cleaning passage with no query tokens.")]}
    )
    result = Harness(retr, slm=_FakeSLM("", ready=False)).run("डोनाल्ड ट्रंप कौन है?")
    assert retr.calls == ["hi"]
    assert result.answer.refused


def test_extractive_fallback_when_model_emits_headline():
    parent = (
        "Donald Trump is an American politician, real-estate developer, author "
        "and television personality who has a net worth of $3.1 billion."
    )
    slm = _FakeSLM("Cabinet Net Worth")
    result = Harness(_FakeRetriever(parent), slm=slm).run("who is donald trump?")
    assert slm.calls == 1
    assert "politician" in result.answer.text.lower() or "developer" in result.answer.text.lower()
    assert "Cabinet" not in result.answer.text
