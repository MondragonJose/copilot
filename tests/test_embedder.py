"""Tests: BgeM3Embedder dimension, L2 normalisation, error handling."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from core.errors import EmbeddingError

pytestmark = pytest.mark.asyncio

DIM = 1024


class _FakeArray:
    """Mimics a numpy array row returned by SentenceTransformer.encode()."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


def _make_mock_encode(dim: int = DIM) -> MagicMock:
    """Return a MagicMock that acts as a SentenceTransformer model.

    The mock's ``encode()`` returns one *unit vector* per input text.
    """

    def encode_side_effect(
        texts: list[str],  # noqa: ARG001
        **kwargs: object,  # noqa: ARG001
    ) -> list[_FakeArray]:
        n = len(texts)
        return [
            _FakeArray(
                # First element = 1.0, rest = 0.0 → unit L2 norm
                [1.0] + [0.0] * (dim - 1),
            )
            for _ in range(n)
        ]

    model = MagicMock()
    model.encode.side_effect = encode_side_effect
    return model


class TestDimension:
    async def test_output_dimension(self) -> None:
        from retrieval.embedder import BgeM3Embedder

        embedder = BgeM3Embedder(model=_make_mock_encode())
        vectors = await embedder.embed(["hello world"])

        assert len(vectors) == 1
        assert len(vectors[0]) == DIM

    async def test_multiple_texts(self) -> None:
        from retrieval.embedder import BgeM3Embedder

        embedder = BgeM3Embedder(model=_make_mock_encode())
        texts = ["first", "second", "third"]
        vectors = await embedder.embed(texts)

        assert len(vectors) == 3
        for v in vectors:
            assert len(v) == DIM

    async def test_empty_input(self) -> None:
        from retrieval.embedder import BgeM3Embedder

        embedder = BgeM3Embedder(model=_make_mock_encode())
        vectors = await embedder.embed([])

        assert vectors == []

    async def test_dim_property(self) -> None:
        from retrieval.embedder import BgeM3Embedder

        embedder = BgeM3Embedder(model=_make_mock_encode())
        assert embedder.dim == DIM


class TestNormalisation:
    async def test_vectors_are_unit_length(self) -> None:
        from retrieval.embedder import BgeM3Embedder

        embedder = BgeM3Embedder(model=_make_mock_encode())
        vectors = await embedder.embed(["a", "b", "c"])

        for v in vectors:
            norm = math.sqrt(sum(x * x for x in v))
            assert norm == pytest.approx(1.0, abs=1e-6), (
                f"Vector not normalised (norm={norm})"
            )

    async def test_no_zero_vectors(self) -> None:
        from retrieval.embedder import BgeM3Embedder

        embedder = BgeM3Embedder(model=_make_mock_encode())
        vectors = await embedder.embed(["test"])

        v = vectors[0]
        norm = math.sqrt(sum(x * x for x in v))
        assert norm > 0, "Embedding should not be a zero vector"


class TestErrorHandling:
    async def test_encoding_error_wraps_embedding_error(self) -> None:
        from retrieval.embedder import BgeM3Embedder

        broken_model = MagicMock()
        broken_model.encode.side_effect = RuntimeError("CUDA out of memory")

        embedder = BgeM3Embedder(model=broken_model)
        with pytest.raises(EmbeddingError, match="CUDA out of memory"):
            await embedder.embed(["trigger error"])

    async def test_embedding_error_is_rc_error(self) -> None:
        from core.errors import RCError
        from retrieval.embedder import BgeM3Embedder

        broken_model = MagicMock()
        broken_model.encode.side_effect = RuntimeError("fail")

        embedder = BgeM3Embedder(model=broken_model)
        with pytest.raises(RCError):
            await embedder.embed(["fail"])
