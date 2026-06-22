from core.errors import (
    EmbeddingError,
    IngestError,
    ParseError,
    RCError,
    RetrievalError,
    VerificationError,
)


def test_rc_error_is_base() -> None:
    assert issubclass(IngestError, RCError)
    assert issubclass(ParseError, RCError)
    assert issubclass(RetrievalError, RCError)
    assert issubclass(EmbeddingError, RCError)
    assert issubclass(VerificationError, RCError)


def test_rc_error_message() -> None:
    e = RCError("something went wrong")
    assert str(e) == "something went wrong"


def test_ingest_error() -> None:
    e = IngestError("failed to ingest")
    assert isinstance(e, RCError)


def test_parse_error_subclass_of_ingest() -> None:
    e = ParseError("parse failed")
    assert isinstance(e, IngestError)


def test_parse_error_message() -> None:
    e = ParseError("PDF not found: /x/y.pdf")
    assert "PDF not found" in str(e)


def test_retrieval_error() -> None:
    e = RetrievalError("connection refused")
    assert isinstance(e, RCError)


def test_embedding_error() -> None:
    e = EmbeddingError("OOM on GPU")
    assert isinstance(e, RCError)


def test_verification_error() -> None:
    e = VerificationError("verifier failed")
    assert isinstance(e, RCError)


def test_errors_are_exceptions() -> None:
    assert issubclass(RCError, Exception)


def test_errors_empty_message() -> None:
    assert str(RCError()) == ""


def test_errors_nested_chaining() -> None:
    inner = ValueError("inner")
    try:
        raise ParseError("outer") from inner
    except ParseError as outer:
        assert isinstance(outer, ParseError)
        assert outer.__cause__ is inner
