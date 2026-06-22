"""Deterministic literal-anchor Verifier — checks quoted_span against chunk text.

``LiteralVerifier`` implements the ``Verifier`` Protocol from
``core.interfaces``.  For each ``Claim`` it looks up the chunk by
``claim.chunk_id`` and tests whether ``claim.quoted_span`` is an exact
substring of the chunk's text.

``TwoLayerVerifier`` adds an entailment judge (via ``LLMClient``) that
produces a 0..1 entailment score.  A claim is ``supported`` only when
both the anchor check passes **and** the entailment score meets a
configurable threshold.
"""

from __future__ import annotations

import asyncio

from core.interfaces import LLMClient
from core.models import Claim, ClaimVerdict
from qa._nli import judge_entailment


class LiteralVerifier:
    """Verify each claim's ``quoted_span`` is a literal substring of its chunk.

    Parameters
    ----------
    chunk_texts
        Mapping ``{chunk_id: full_text}`` for every chunk that may be cited.
    """

    def __init__(self, chunk_texts: dict[str, str]) -> None:
        self._chunk_texts = chunk_texts

    def verify(self, claims: list[Claim]) -> list[ClaimVerdict]:
        """Return a verdict for each claim in *claims*."""
        return [self._verify_one(c) for c in claims]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _verify_one(self, claim: Claim) -> ClaimVerdict:
        chunk_text = self._chunk_texts.get(claim.chunk_id)
        if chunk_text is None:
            return ClaimVerdict(
                claim=claim,
                supported=False,
                score=0.0,
                reason="chunk_not_found",
            )

        if claim.quoted_span not in chunk_text:
            return ClaimVerdict(
                claim=claim,
                supported=False,
                score=0.0,
                reason="span_not_found",
            )

        return ClaimVerdict(
            claim=claim,
            supported=True,
            score=1.0,
            reason="",
        )


class TwoLayerVerifier:
    """Two-layer claim verifier: literal anchor check + entailment judge.

    Layer 1 — literal anchor
        Verifies that ``claim.quoted_span`` is an exact substring of the
        chunk identified by ``claim.chunk_id``.

    Layer 2 — entailment score
        Uses an ``LLMClient`` to judge whether the chunk text entails the
        claim text, producing a 0..1 score.

    A claim is ``supported`` only when **both** checks pass: the anchor
    check succeeds **and** the entailment score ≥ *threshold*.

    Parameters
    ----------
    chunk_texts
        Mapping ``{chunk_id: full_text}`` for every chunk that may be cited.
    llm
        LLM client used as the NLI entailment judge.
    threshold
        Minimum entailment score required for a claim to be supported.
        Must be in ``[0, 1]``.  Default 0.5.
    """

    def __init__(
        self,
        chunk_texts: dict[str, str],
        llm: LLMClient,
        *,
        threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self._chunk_texts = chunk_texts
        self._llm = llm
        self._threshold = threshold

    def verify(self, claims: list[Claim]) -> list[ClaimVerdict]:
        """Return a verdict for each claim in *claims*."""
        return asyncio.run(self._verify_async(claims))

    async def _verify_async(self, claims: list[Claim]) -> list[ClaimVerdict]:
        verdicts: list[ClaimVerdict] = []
        for claim in claims:
            verdicts.append(await self._verify_one(claim))
        return verdicts

    async def _verify_one(self, claim: Claim) -> ClaimVerdict:
        # --- Layer 1: literal anchor ----------------------------------------
        chunk_text = self._chunk_texts.get(claim.chunk_id)
        if chunk_text is None:
            return ClaimVerdict(
                claim=claim,
                supported=False,
                score=0.0,
                reason="chunk_not_found",
            )

        if claim.quoted_span not in chunk_text:
            return ClaimVerdict(
                claim=claim,
                supported=False,
                score=0.0,
                reason="span_not_found",
            )

        # --- Layer 2: entailment judge ---------------------------------------
        entailment_score = await judge_entailment(
            self._llm, chunk_text, claim.text,
        )

        if entailment_score >= self._threshold:
            return ClaimVerdict(
                claim=claim,
                supported=True,
                score=entailment_score,
                reason="",
            )

        return ClaimVerdict(
            claim=claim,
            supported=False,
            score=entailment_score,
            reason="entailment_below_threshold",
        )
