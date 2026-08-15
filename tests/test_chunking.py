from voice_rag.chunking.ensemble import EnsembleChunker
from voice_rag.types import ChunkStrategy


PASSAGE = (
    "The Manhattan Project was a research and development undertaking during "
    "World War II that produced the first nuclear weapons. It was led by the "
    "United States with the support of the United Kingdom and Canada. "
    "The project succeeded in detonating the first atomic bomb in July 1945."
)


def test_ensemble_emits_multiple_strategies():
    chunker = EnsembleChunker()
    chunks = chunker.chunk_one(PASSAGE, "p1", "en", query_types=["DESCRIPTION"], is_gold=True)
    strategies = {c.strategy for c in chunks}
    assert ChunkStrategy.PASSAGE in strategies
    assert ChunkStrategy.SENTENCE_WINDOW in strategies
    assert ChunkStrategy.RECURSIVE in strategies
    assert ChunkStrategy.PROPOSITION in strategies
    assert ChunkStrategy.HIERARCHICAL_CHILD in strategies
    assert len(chunks) >= 6
    assert all(c.parent_id == "p1" for c in chunks)
    assert any(c.prev_ctx or c.next_ctx for c in chunks)


def test_overlap_windows_share_text():
    chunker = EnsembleChunker()
    chunks = [c for c in chunker.chunk_one(PASSAGE, "p1", "en") if c.strategy == ChunkStrategy.SENTENCE_WINDOW]
    if len(chunks) >= 2:
        a, b = chunks[0].text, chunks[1].text
        shared = set(a.split()) & set(b.split())
        assert shared
