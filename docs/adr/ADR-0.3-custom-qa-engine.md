# ADR-0.3: Custom Retrieve-Then-Generate QA Engine

**Status:** Accepted

---

## Context

The MVP blueprint (RESEARCH_COPILOT_PHASE_0.1.md) specifies a QA engine based on
a custom retrieve-then-generate pattern (S4). An earlier planning document
referenced wrapping PaperQA2 (`langchain-ai/paper-qa`) as the answer generation
backend.

PaperQA2 is a library that retrieves chunks from a user-provided corpus and
generates answers with citations via an LLM. At first glance it maps well to
the project's requirements — grounded citation verification.

Key requirements that PaperQA2 does not fully satisfy:

- **Two-layer verification** — every claim must pass a literal anchor check
  (quoted_span must be a substring of the chunk text) AND an LLM entailment
  judge producing a 0..1 score. PaperQA2 provides no built-in span verification.
- **Drop/retry/degrade/abstain policy** — claims that fail verification trigger
  a retry with doubled context, degradation to partial credit, or full abstention.
  This policy chain is custom per Blueprint §5.
- **Channel metadata** — each scored chunk must carry a `channel` tag
  (`dense`, `lexical`, `hybrid`) for downstream metric decomposition.
- **No LLM framework lock-in** — the engine must support OpenAI and Ollama
  interchangeably without depending on langchain or similar.

---

## Decision

1. **Build a custom QA engine** in `packages/qa/engine.py` implementing
   retrieve-then-generate with inline verification, without wrapping PaperQA2.

2. **Expose LLM interaction through the `LLMClient` Protocol**
   (`core/interfaces.py:76`) so providers can be swapped without changing
   the engine.

3. **Implement the verification policy inline** (drop / retry / degrade /
   abstain per Blueprint §5) rather than delegating to any external library.

---

## Consequences

- Full control over the verification pipeline — every branch is testable
  in isolation (see `TestDropBranch`, `TestRetryBranch`, `TestDegradeBranch`,
  `TestAbstainBranch` in `tests/qa/test_engine.py`).
- No dependency on PaperQA2 or its transitive dependencies (langchain,
  numpy, etc.), keeping the install footprint smaller.
- More code to maintain than a library wrapper — approximately 400 lines
  for the engine + verifier + NLI judge.
- Existing tests (605 unit tests) cover the custom engine end-to-end.

---

## Reversibility

**Easily reversible** — the `LLMClient` Protocol is a thin abstraction over
`generate(prompt, system)`. Switching to PaperQA2 would require writing an
adapter that maps `LLMClient` calls to PaperQA2's API, but is otherwise
straightforward.

---

## Alternatives Considered

- **Wrap PaperQA2 as the generator, then post-process** — rejected because
  the two-layer verification, policy branching, and channel metadata would
  require patching PaperQA2 internals. The abstraction would leak.
- **Use LangChain Expression Language (LCEL)** — rejected to avoid
  framework lock-in and reduce dependency surface.
- **Delegate verification to a separate NLI-as-a-service** — rejected for
  v0.1; the `judge_entailment` function (`packages/qa/_nli.py`) is simple
  enough to keep in-repo.

---

## Blueprint References

- **S4** (Motor QA) — Custom retrieve-then-generate — implemented as
  `packages/qa/engine.py:QAEngine`.
- **S5** (LLM dev) — `LLMClient` Protocol implemented by
  `packages/qa/llm.py:LLMProvider`.
- **§5** (Verification policy) — Implemented in `QAEngine._apply_policy`:
  `packages/qa/engine.py:136`.
