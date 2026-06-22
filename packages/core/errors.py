"""Domain exception hierarchy for Research Copilot.

Every operation that crosses an I/O boundary raises a typed exception
so callers can handle failures without catching bare Exception.
"""


class RCError(Exception):
    """Base exception for all Research Copilot domain errors."""


class IngestError(RCError):
    """Raised when the ingest pipeline fails."""


class ParseError(IngestError):
    """Raised when a PDF parser cannot extract content from a document."""


class RetrievalError(RCError):
    """Raised when the retrieval store (pgvector / Qdrant) fails."""


class EmbeddingError(RCError):
    """Raised when the embedder cannot produce vectors."""


class VerificationError(RCError):
    """Raised when the verifier cannot check a claim."""
