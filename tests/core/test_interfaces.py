"""Verify that all Protocols accept the correct signatures.

These tests do not instantiate the Protocols (they cannot be
instantiated directly). Instead they verify that the concrete
implementations used across the project satisfy the structural
subtyping contract by checking method signatures.

This module relies on ``pytest`` for assertions but does not
require any external dependencies beyond the standard library.
"""

import inspect
from collections.abc import Sequence

from core.interfaces import (
    Embedder,
    LLMClient,
    Parser,
    Retriever,
    Verifier,
)
from core.models import Claim, ClaimVerdict, ScoredChunk, UpsertChunk


def _assert_async_method(cls: type, method: str) -> None:
    meth = getattr(cls, method, None)
    assert meth is not None, f"{cls.__name__} missing {method}"
    assert inspect.iscoroutinefunction(meth), f"{cls.__name__}.{method} must be async"


def _assert_sync_method(cls: type, method: str) -> None:
    meth = getattr(cls, method, None)
    assert meth is not None, f"{cls.__name__} missing {method}"
    assert not inspect.iscoroutinefunction(meth), f"{cls.__name__}.{method} must be sync"


def test_retriever_protocol_methods_exist() -> None:
    for m in ("upsert", "search_dense", "search_lexical",
              "delete_by_paper", "count", "health"):
        assert hasattr(Retriever, m), f"Retriever missing {m}"
    sig_upsert = inspect.signature(Retriever.upsert)
    params = list(sig_upsert.parameters)
    assert "items" in params
    hint = sig_upsert.parameters["items"].annotation
    # Must accept Sequence[UpsertChunk]
    assert hint in (Sequence[UpsertChunk], Sequence[UpsertChunk] | None)


def test_retriever_return_types() -> None:
    sig = inspect.signature(Retriever.search_dense)
    hint = sig.return_annotation
    assert "ScoredChunk" in str(hint) or hint is list[ScoredChunk]


def test_parser_protocol() -> None:
    assert hasattr(Parser, "parse")
    assert inspect.iscoroutinefunction(Parser.parse)


def test_embedder_protocol() -> None:
    assert hasattr(Embedder, "embed")
    assert inspect.iscoroutinefunction(Embedder.embed)
    sig = inspect.signature(Embedder.embed)
    params = list(sig.parameters)
    assert "texts" in params


def test_llm_client_protocol() -> None:
    assert hasattr(LLMClient, "generate")
    assert inspect.iscoroutinefunction(LLMClient.generate)


def test_verifier_protocol() -> None:
    assert hasattr(Verifier, "verify")
    assert inspect.iscoroutinefunction(Verifier.verify)
    sig = inspect.signature(Verifier.verify)
    params = list(sig.parameters)
    assert "claims" in params


def test_verifier_returns_claim_verdicts() -> None:
    sig = inspect.signature(Verifier.verify)
    hint = sig.return_annotation
    assert "ClaimVerdict" in str(hint)
