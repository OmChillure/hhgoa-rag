from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ChunkStrategy(str, Enum):
    PASSAGE = "passage"
    SENTENCE_WINDOW = "sentence_window"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    PROPOSITION = "proposition"
    HIERARCHICAL_CHILD = "hierarchical_child"


class QueryType(str, Enum):
    DESCRIPTION = "DESCRIPTION"
    NUMERIC = "NUMERIC"
    ENTITY = "ENTITY"
    LOCATION = "LOCATION"
    PERSON = "PERSON"
    UNKNOWN = "UNKNOWN"


class Language(str, Enum):
    EN = "en"
    HI = "hi"
    OTHER = "other"


class Chunk(BaseModel):
    chunk_id: str
    parent_id: str
    strategy: ChunkStrategy
    text: str
    language: str
    query_types: list[str] = Field(default_factory=list)
    source_query_ids: list[int] = Field(default_factory=list)
    is_gold: bool = False
    position: int = 0
    n_chars: int = 0
    prev_ctx: str = ""
    next_ctx: str = ""


class Hit(BaseModel):
    chunk: Chunk
    score: float
    rank: int
    origin: str
    parent_text: str = ""


class GuardDecision(BaseModel):
    allowed: bool
    stage: str
    reason: str
    categories: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class Span(BaseModel):
    text: str
    parent_id: str
    start: int
    end: int
    score: float


class Answer(BaseModel):
    text: str
    mode: str
    confidence: float
    spans: list[Span] = Field(default_factory=list)
    citations: list[Hit] = Field(default_factory=list)
    grounded: bool = False
    coverage: float = 0.0
    refused: bool = False
    refusal_reason: str = ""


class StageTiming(BaseModel):
    name: str
    ms: float


class PipelineResult(BaseModel):
    query: str
    transcript: str | None = None
    detected_language: str = "en"
    query_type: str = "UNKNOWN"
    answer: Answer
    guardrails: list[GuardDecision] = Field(default_factory=list)
    hits: list[Hit] = Field(default_factory=list)
    timings: list[StageTiming] = Field(default_factory=list)
    total_ms: float = 0.0
    sla_ok: bool = False
    harness_trace: list[dict[str, Any]] = Field(default_factory=list)
    stt_ms: float | None = None
