"""BGE-M3 embedder implementing the Embedder Protocol.

Wraps ``sentence-transformers`` model ``BAAI/bge-m3`` with L2 normalisation,
configurable batch size (``EMBEDDING_BATCH_SIZE`` env / constructor arg),
and thread-pool offloading for async compatibility.

Reference machine ........... Apple M4 Pro (14-core CPU, 24 GB unified memory)
  ~800 tok/s ................ MPS (Metal Performance Shaders)
  ~400 tok/s ................ CPU only
Throughput depends on hardware, batch size, and sequence length.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Sequence
from typing import Any

from core._constants import DEFAULT_BATCH_SIZE
from core.errors import EmbeddingError

logger = logging.getLogger(__name__)

DIM = 1024


class BgeM3Embedder:
    """BGE-M3 embedder with L2 normalisation and async offloading."""

    def __init__(
        self,
        batch_size: int | None = None,
        model: Any | None = None,
    ) -> None:
        self._batch_size = batch_size or int(
            os.environ.get("EMBEDDING_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)),
        )
        self._model: Any = model
        self._dim = DIM

    # ------------------------------------------------------------------
    # Embedder Protocol
    # ------------------------------------------------------------------

    async def embed(self, texts: list[str]) -> list[Sequence[float]]:
        """Embed a batch of texts and return L2-normalised 1024-dim vectors."""
        if not texts:
            return []

        model = self._get_model()
        loop = asyncio.get_running_loop()

        try:
            embeddings = await loop.run_in_executor(
                None,
                lambda: model.encode(
                    texts,
                    batch_size=self._batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
            )
        except Exception as exc:
            raise EmbeddingError(f"BGE-M3 encoding failed: {exc}") from exc

        return [emb.tolist() for emb in embeddings]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> Any:
        """Lazy-load the SentenceTransformer model (deferred import)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

            logger.info(
                "Loading BGE-M3 embedder (dim=%d, batch_size=%d). "
                "Reference machine: Apple M4 Pro (14-core CPU, 24 GB unified memory) "
                "— ~800 tok/s on MPS, ~400 tok/s on CPU.",
                self._dim,
                self._batch_size,
            )
            self._model = SentenceTransformer("BAAI/bge-m3")
        return self._model

    @property
    def dim(self) -> int:
        """Embedding dimension (1024 for BGE-M3)."""
        return self._dim
