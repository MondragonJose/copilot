"""Retrieve-then-generate QA engine — wraps our Retriever + LLMClient.

Produces a ``QAResult`` whose ``claims`` each carry a ``chunk_id`` and
a ``quoted_span`` recovered from the chunk text.

Policy (Blueprint §5)
---------------------
1. **Drop** — fabricated claims (span not in chunk) are removed.
2. **Retry** — anchored-but-low-entailment claims trigger one retry with
   doubled retrieval ``k`` and a fresh generation.
3. **Degrade** — after retry, any still-low claim is kept as
   ``supported=False, reason='insufficient_evidence'``.
4. **Abstain** — if no claim reaches ``supported=True`` the answer is
   ``"No hay soporte suficiente"`` with ``answerable=False``.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence

_RETRY_K_MULTIPLIER = 2

from core.interfaces import Embedder, LLMClient, Retriever
from core.models import Claim, ClaimVerdict, QAResult, ScoredChunk
from qa._nli import judge_entailment

_SYSTEM_PROMPT = """You are a helpful research assistant. Answer the user's \
question based solely on the provided context chunks.

Each chunk is numbered: [1], [2], etc.  When you make a factual claim, \
cite the supporting chunk inline like [1] or [2-3].

After your answer, add a ## Citations section with one line per citation:

## Citations
[1] chunk: CHUNK_ID_HERE quote: "The exact text from chunk [1] that supports \
your claim."
[2] chunk: CHUNK_ID_HERE quote: "The exact text from chunk [2] that supports \
your claim."

Use the chunk_id shown in the context, not the number.
Every claim must have a corresponding citations entry.
If the question cannot be answered from the context, say so and leave the \
## Citations section empty."""




class QAEngine:
    """Retrieve-then-generate QA engine.

    Parameters
    ----------
    retriever
        Used to fetch relevant chunks for the question.
    llm
        Used to generate the answer with citations.
    embedder
        Used to embed the question for dense retrieval.
    k
        Default number of chunks to retrieve.
    verifier_threshold
        Minimum entailment score for a claim to be considered supported.
        Must be in ``[0, 1]``.  Default 0.5.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        embedder: Embedder,
        *,
        k: int = 10,
        verifier_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= verifier_threshold <= 1.0:
            raise ValueError(
                "verifier_threshold must be in [0, 1], "
                f"got {verifier_threshold}",
            )
        self._retriever = retriever
        self._llm = llm
        self._embedder = embedder
        self._k = k
        self._verifier_threshold = verifier_threshold

    async def answer(
        self,
        question: str,
        k: int | None = None,
    ) -> QAResult:
        """Answer *question* and apply the Blueprint §5 verification policy.

        Returns a ``QAResult`` after running retrieval → generation →
        two-layer verification → drop / retry / degrade / abstain.
        """
        if not question.strip():
            return QAResult(
                question=question,
                answer=None,
                claims=[],
                verdicts=[],
                answerable=False,
            )

        top_k = k or self._k

        vectors = await self._embedder.embed([question])
        scored = await self._retriever.search_dense(vectors[0], k=top_k)  # type: ignore[misc]

        if not scored:
            return QAResult(
                question=question,
                answer=None,
                claims=[],
                verdicts=[],
                answerable=False,
            )

        answer_text, claims = await self._generate(question, scored)

        return await self._apply_policy(
            question=question,
            answer_text=answer_text,
            claims=claims,
            scored=scored,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def _generate(
        self,
        question: str,
        scored: Sequence[ScoredChunk],
    ) -> tuple[str, list[Claim]]:
        """Run the LLM and parse citations from its answer."""
        context = _format_context(scored)
        prompt = _build_prompt(question, context)
        answer_text = await self._llm.generate(prompt, system=_SYSTEM_PROMPT)
        chunk_map = _build_chunk_map(scored)
        claims = _parse_citations(answer_text, chunk_map)
        return answer_text, claims

    # ------------------------------------------------------------------
    # Verification — anchor + entailment judge
    # ------------------------------------------------------------------

    async def _verify_claims(
        self,
        claims: list[Claim],
        chunk_texts: dict[str, str],
    ) -> list[ClaimVerdict]:
        """Two-layer verification for every claim.

        Layer 1 — literal anchor: ``quoted_span`` must be an exact substring
        of the chunk text.
        Layer 2 — entailment: LLM-based NLI judge produces a 0..1 score.
        """
        return list(await asyncio.gather(*[
            self._verify_one(claim, chunk_texts) for claim in claims
        ]))

    async def _verify_one(
        self,
        claim: Claim,
        chunk_texts: dict[str, str],
    ) -> ClaimVerdict:
        chunk_text = chunk_texts.get(claim.chunk_id)
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

        entailment_score = await judge_entailment(
            self._llm, chunk_text, claim.text,
        )

        if entailment_score >= self._verifier_threshold:
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

    # ------------------------------------------------------------------
    # Blueprint §5 policy
    # ------------------------------------------------------------------

    async def _apply_policy(
        self,
        question: str,
        answer_text: str,
        claims: list[Claim],
        scored: Sequence[ScoredChunk],
        top_k: int,
    ) -> QAResult:
        """Apply drop → retry → degrade → abstain on *claims*."""
        chunk_texts = {sc.chunk.chunk_id: sc.chunk.text for sc in scored}
        verdicts = await self._verify_claims(claims, chunk_texts)

        surviving = self._drop_fabricated(claims, verdicts)
        answer_text, claims, verdicts = await self._retry_if_needed(
            question, surviving, answer_text, claims, verdicts, top_k,
        )
        final_claims, final_verdicts = self._degrade_low_claims(claims, verdicts)
        return self._finalize(question, answer_text, final_claims, final_verdicts)

    @staticmethod
    def _drop_fabricated(
        claims: list[Claim],
        verdicts: list[ClaimVerdict],
    ) -> list[tuple[Claim, ClaimVerdict]]:
        return [
            (c, v)
            for c, v in zip(claims, verdicts)
            if v.reason not in ("span_not_found", "chunk_not_found")
        ]

    async def _retry_if_needed(
        self,
        question: str,
        surviving: list[tuple[Claim, ClaimVerdict]],
        answer_text: str,
        claims: list[Claim],
        verdicts: list[ClaimVerdict],
        top_k: int,
    ) -> tuple[str, list[Claim], list[ClaimVerdict]]:
        needs_retry = any(
            v.reason == "entailment_below_threshold" for _, v in surviving
        )
        if not needs_retry:
            return answer_text, claims, verdicts

        doubled_k = top_k * _RETRY_K_MULTIPLIER
        vectors = await self._embedder.embed([question])
        scored2 = await self._retriever.search_dense(
            vectors[0], k=doubled_k,  # type: ignore[misc]
        )
        if not scored2:
            claims2, verdicts2 = self._split_claim_verdict_pairs(
                surviving,
            ) if surviving else ([], [])
            return answer_text, claims2, verdicts2

        answer_text2, claims2 = await self._generate(question, scored2)
        chunk_texts2 = {
            sc.chunk.chunk_id: sc.chunk.text for sc in scored2
        }
        verdicts2 = await self._verify_claims(claims2, chunk_texts2)
        return answer_text2, claims2, verdicts2

    @staticmethod
    def _degrade_low_claims(
        claims: list[Claim],
        verdicts: list[ClaimVerdict],
    ) -> tuple[list[Claim], list[ClaimVerdict]]:
        final_claims: list[Claim] = []
        final_verdicts: list[ClaimVerdict] = []
        for c, v in zip(claims, verdicts):
            if v.reason in ("span_not_found", "chunk_not_found"):
                continue
            if v.reason == "entailment_below_threshold":
                v = ClaimVerdict(
                    claim=c,
                    supported=False,
                    score=v.score,
                    reason="insufficient_evidence",
                )
            final_claims.append(c)
            final_verdicts.append(v)
        return final_claims, final_verdicts

    @staticmethod
    def _finalize(
        question: str,
        answer_text: str,
        claims: list[Claim],
        verdicts: list[ClaimVerdict],
    ) -> QAResult:
        any_supported = any(v.supported for v in verdicts)
        if not any_supported:
            return QAResult(
                question=question,
                answer="No hay soporte suficiente",
                claims=claims,
                verdicts=verdicts,
                answerable=False,
            )
        return QAResult(
            question=question,
            answer=answer_text,
            claims=claims,
            verdicts=verdicts,
            answerable=True,
        )

    @staticmethod
    def _split_claim_verdict_pairs(
        pairs: list[tuple[Claim, ClaimVerdict]],
    ) -> tuple[list[Claim], list[ClaimVerdict]]:
        claims, verdicts = zip(*pairs) if pairs else ([], [])
        return list(claims), list(verdicts)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _format_context(scored: Sequence[ScoredChunk]) -> str:
    """Format retrieved chunks into a numbered context block for the LLM."""
    lines: list[str] = []
    for i, sc in enumerate(scored, start=1):
        ref = sc.chunk
        section_tag = f" (section: {ref.section})" if ref.section else ""
        lines.append(
            f"[{i}] chunk_id: {ref.chunk_id}{section_tag}\n"
            f"     text: {ref.text}",
        )
    return "\n\n".join(lines)


def _build_prompt(question: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}"


# ---------------------------------------------------------------------------
# Claim extraction — parse ## Citations section
# ---------------------------------------------------------------------------


def _build_chunk_map(
    scored: Sequence[ScoredChunk],
) -> dict[int, str]:
    """Map citation number → chunk_id for every scored chunk."""
    return {i: sc.chunk.chunk_id for i, sc in enumerate(scored, start=1)}


def _parse_citations(
    answer: str,
    chunk_map: dict[int, str],
) -> list[Claim]:
    """Parse the ``## Citations`` section at the end of the answer.

    Expected format per line::

        [N] chunk: <uuid> quote: "quoted text"
    """
    claims: list[Claim] = []

    citations_match = re.search(
        r"## Citations\s*\n(.*)",
        answer,
        flags=re.DOTALL,
    )
    if not citations_match:
        return claims

    body = citations_match.group(1)

    pattern = re.compile(
        r"^\[(\d+)\]\s+chunk:\s*(\S+)\s+quote:\s*\"(.*)\"\s*$",
        flags=re.MULTILINE,
    )

    for match in pattern.finditer(body):
        num = int(match.group(1))
        chunk_id = match.group(2)
        quote = match.group(3).strip()

        if num in chunk_map:
            claims.append(Claim(
                text=quote,
                chunk_id=chunk_id,
                quoted_span=quote,
            ))

    return claims
