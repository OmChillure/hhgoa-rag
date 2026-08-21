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
    assert result.answer.mode == "extractive"
    assert "Paris" in result.answer.text
    assert "Berlin" not in result.answer.text


def test_extractive_when_slm_missing():
    parent = "Paris is the capital and most populous city of France."
    slm = _FakeSLM("", ready=False)
    result = Harness(_FakeRetriever(parent), slm=slm).run("what is the capital of france?")
    assert slm.calls == 0
    assert result.answer.mode == "extractive"
    assert "Paris" in result.answer.text
