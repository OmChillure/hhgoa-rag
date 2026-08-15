from voice_rag.generation.extractive import extract
from voice_rag.types import Chunk, ChunkStrategy, Hit


def test_extractive_picks_supported_span():
    parent = (
        "Paris is the capital and most populous city of France. "
        "It is located on the river Seine in the north of the country."
    )
    ch = Chunk(
        chunk_id="c",
        parent_id="p",
        strategy=ChunkStrategy.PASSAGE,
        text=parent,
        language="en",
    )
    hits = [Hit(chunk=ch, score=0.4, rank=0, origin="bm25", parent_text=parent)]
    answer, spans, conf = extract("what is the capital of france?", hits, "LOCATION")
    assert "Paris" in answer
    assert spans
    assert conf > 0
