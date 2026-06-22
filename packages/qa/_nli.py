"""Shared NLI (natural language inference) judge.

Used by both the inline ``QAEngine`` verification (``qa/engine.py``) and
the standalone ``TwoLayerVerifier`` (``qa/verifier.py``).
"""

from core._utils import clamp_score
from core.interfaces import LLMClient

NLI_SYSTEM_PROMPT = (
    "You are an NLI (natural language inference) judge. "
    "Given a context passage and a claim, rate how well the context "
    "supports (entails) the claim on a scale from 0.0 to 1.0. "
    "Return ONLY a single float number with no other text.\n\n"
    "0.0 = completely contradictory / the context refutes the claim\n"
    "0.5 = neutral / unrelated / not enough information\n"
    "1.0 = clearly entailed / the context directly supports the claim"
)


async def judge_entailment(
    llm: LLMClient,
    context: str,
    claim_text: str,
) -> float:
    """Ask *llm* to rate how well *context* entails *claim_text*.

    Returns a clamped float in ``[0.0, 1.0]``.  Returns ``0.0`` on
    response-parsing failure.
    """
    prompt = (
        f"Context: {context}\n\n"
        f"Claim: {claim_text}\n\n"
        f"Score (0.0 to 1.0):"
    )
    response = await llm.generate(prompt, system=NLI_SYSTEM_PROMPT)
    try:
        return clamp_score(float(response.strip()))
    except (ValueError, TypeError):
        return 0.0
