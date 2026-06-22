"""Tests for BgeM3Embedder — model is mocked; no real inference."""

import math
from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.errors import EmbeddingError, RCError
from retrieval.embedder import BgeM3Embedder


class MockNumpyArray:
    """Minimal mock for np.ndarray.tolist()."""
    def __init__(self, data: list[float]) -> None:
        self._data = data

    def tolist(self) -> list[float]:
        return self._data


@pytest.fixture
def mock_sentence_transformer() -> MagicMock:
    model = MagicMock()
    model.encode = MagicMock()
    return model


class TestBgeM3Embedder:
    def test_default_dim(self) -> None:
        e = BgeM3Embedder()
        assert e.dim == 1024

    def test_default_batch_size(self) -> None:
        e = BgeM3Embedder()
        assert e._batch_size == 32

    def test_custom_batch_size(self) -> None:
        e = BgeM3Embedder(batch_size=8)
        assert e._batch_size == 8

    @pytest.mark.asyncio
    async def test_embed_empty_list(self) -> None:
        e = BgeM3Embedder()
        assert await e.embed([]) == []

    @pytest.mark.asyncio
    async def test_embed_success(self, mock_sentence_transformer: MagicMock) -> None:
        mock_sentence_transformer.encode.return_value = [
            MockNumpyArray([0.1, 0.2]),
            MockNumpyArray([0.3, 0.4]),
        ]
        e = BgeM3Embedder(model=mock_sentence_transformer)
        vectors = await e.embed(["hello", "world"])
        assert len(vectors) == 2
        assert vectors[0] == [0.1, 0.2]
        assert vectors[1] == [0.3, 0.4]

    @pytest.mark.asyncio
    async def test_embed_single_text(self, mock_sentence_transformer: MagicMock) -> None:
        mock_sentence_transformer.encode.return_value = [
            MockNumpyArray([0.5]),
        ]
        e = BgeM3Embedder(model=mock_sentence_transformer)
        vectors = await e.embed(["single"])
        assert len(vectors) == 1
        assert vectors[0] == [0.5]

    @pytest.mark.asyncio
    async def test_embed_raises_embedding_error(
        self, mock_sentence_transformer: MagicMock,
    ) -> None:
        mock_sentence_transformer.encode.side_effect = RuntimeError("OOM")
        e = BgeM3Embedder(model=mock_sentence_transformer)
        with pytest.raises(EmbeddingError, match="BGE-M3 encoding failed"):
            await e.embed(["test"])

    @pytest.mark.asyncio
    async def test_embedding_error_is_rc_error(
        self, mock_sentence_transformer: MagicMock,
    ) -> None:
        mock_sentence_transformer.encode.side_effect = RuntimeError("fail")
        e = BgeM3Embedder(model=mock_sentence_transformer)
        with pytest.raises(RCError):
            await e.embed(["fail"])

    @pytest.mark.asyncio
    async def test_vectors_are_unit_length(
        self, mock_sentence_transformer: MagicMock,
    ) -> None:
        mock_sentence_transformer.encode.return_value = [
            MockNumpyArray([1.0, 0.0]),
            MockNumpyArray([0.0, 1.0]),
        ]
        e = BgeM3Embedder(model=mock_sentence_transformer)
        vectors = await e.embed(["a", "b"])
        for v in vectors:
            norm = math.sqrt(sum(x * x for x in v))
            assert norm == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_no_zero_vectors(
        self, mock_sentence_transformer: MagicMock,
    ) -> None:
        mock_sentence_transformer.encode.return_value = [
            MockNumpyArray([1.0, 0.0]),
        ]
        e = BgeM3Embedder(model=mock_sentence_transformer)
        vectors = await e.embed(["test"])
        v = vectors[0]
        norm = math.sqrt(sum(x * x for x in v))
        assert norm > 0

    @pytest.mark.asyncio
    async def test_embed_unicode(self, mock_sentence_transformer: MagicMock) -> None:
        mock_sentence_transformer.encode.return_value = [
            MockNumpyArray([0.1]),
        ]
        e = BgeM3Embedder(model=mock_sentence_transformer)
        vectors = await e.embed(["über cool"])
        assert vectors == [[0.1]]

    @pytest.mark.asyncio
    async def test_embed_large_batch(self, mock_sentence_transformer: MagicMock) -> None:
        texts = [str(i) for i in range(100)]
        mock_sentence_transformer.encode.return_value = [
            MockNumpyArray([0.1]) for _ in texts
        ]
        e = BgeM3Embedder(model=mock_sentence_transformer, batch_size=10)
        vectors = await e.embed(texts)
        assert len(vectors) == 100

    def test_get_model_lazy_loads(self) -> None:
        e = BgeM3Embedder()
        assert e._model is None
        mock_model = MagicMock()
        with patch.object(e, "_get_model", return_value=mock_model):
            model = e._get_model()
            assert model is mock_model
