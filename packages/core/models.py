from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Paper:
    id: str
    doi: str | None
    title: str
    authors: list[dict[str, str]]
    year: int | None
    venue: str | None
    abstract: str | None
    source: str
    open_access: bool | None
    pdf_path: str | None
    grobid_tei: str | None
    meta: dict[str, Any]


@dataclass(frozen=True)
class Chunk:
    id: str
    paper_id: str
    ordinal: int
    section: str | None
    text: str
    char_start: int | None
    char_end: int | None
    page: int | None
    token_count: int | None
    content_hash: str


@dataclass(frozen=True)
class ChunkRef:
    chunk_id: str
    paper_id: str
    text: str
    section: str | None
    page: int | None
    char_start: int | None
    char_end: int | None


@dataclass(frozen=True)
class ScoredChunk:
    chunk: ChunkRef
    score: float
    channel: str


@dataclass(frozen=True)
class UpsertChunk:
    chunk_id: str
    paper_id: str
    text: str
    vector: Sequence[float]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Claim:
    text: str
    chunk_id: str
    quoted_span: str


@dataclass(frozen=True)
class ClaimVerdict:
    claim: Claim
    supported: bool
    score: float
    reason: str


@dataclass(frozen=True)
class QAResult:
    question: str
    answer: str | None
    claims: list[Claim]
    verdicts: list[ClaimVerdict]
    answerable: bool
