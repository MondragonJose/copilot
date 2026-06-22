"""LLM provider — OpenAI-compatible and Ollama behind the LLMClient Protocol.

Usage::

    client = LLMProvider()
    answer = await client.generate("What is RLHF?",
                                   system="You are a helpful assistant.")

Provider selected via env var ``LLM_PROVIDER`` (default ``"openai"``).
``"ollama"`` is opt-in and routes to a local Ollama instance.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import httpx

from core._utils import exponential_backoff
from core.interfaces import LLMClient


class LLMError(RuntimeError):
    """Raised when the LLM provider cannot generate a completion."""


_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class LLMConfig:
    """Read-only configuration sourced from environment variables."""

    provider: str = "openai"
    api_key: str | None = None
    base_url: str | None = None
    model: str = "gpt-4o-mini"
    timeout: float = 30.0
    max_retries: int = 3
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(
            provider=os.environ.get("LLM_PROVIDER", "openai"),
            api_key=os.environ.get("LLM_API_KEY"),
            base_url=os.environ.get("LLM_BASE_URL"),
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            timeout=float(os.environ.get("LLM_TIMEOUT", "30.0")),
            max_retries=int(os.environ.get("LLM_MAX_RETRIES", "3")),
            ollama_base_url=os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434",
            ),
            ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
        )


class LLMProvider(LLMClient):
    """LLMClient implementation for OpenAI-compatible APIs and Ollama.

    Parameters
    ----------
    config
        Configuration object.  Defaults to ``LLMConfig.from_env()``.
    client
        Optional ``httpx.AsyncClient`` for testing.  When omitted a default
        client is created lazily.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config or LLMConfig.from_env()
        self._client = client

    async def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a completion for the given prompt.

        Retries on transient failures (429, 5xx, timeout) with exponential
        backoff up to ``max_retries`` attempts.

        Raises
        ------
        LLMError
            If all retries are exhausted or a non-retryable error occurs.
        """
        last_exc: Exception | None = None

        for attempt in range(1, self._config.max_retries + 1):
            try:
                return await self._try_generate(prompt, system)
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                if attempt < self._config.max_retries and _is_retryable(exc):
                    await asyncio.sleep(exponential_backoff(attempt, base_delay=2.0))
                    last_exc = exc
                    continue
                raise LLMError(str(exc)) from exc

        raise LLMError(str(last_exc))

    # ------------------------------------------------------------------
    # Internal routing
    # ------------------------------------------------------------------

    async def _try_generate(self, prompt: str, system: str | None) -> str:
        if self._config.provider == "ollama":
            return await self._call_ollama(prompt, system)
        return await self._call_openai(prompt, system)

    # ------------------------------------------------------------------
    # OpenAI-compatible /v1/chat/completions
    # ------------------------------------------------------------------

    async def _call_openai(self, prompt: str, system: str | None) -> str:
        client = await self._get_client()
        messages = _build_messages(prompt, system)

        base_url = (self._config.base_url or "https://api.openai.com/v1").rstrip("/")
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        key = self._config.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"

        resp = await client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={"model": self._config.model, "messages": messages},
        )
        resp.raise_for_status()
        data = resp.json()
        content: str = data["choices"][0]["message"]["content"]
        return content

    # ------------------------------------------------------------------
    # Ollama /api/chat
    # ------------------------------------------------------------------

    async def _call_ollama(self, prompt: str, system: str | None) -> str:
        client = await self._get_client()
        messages = _build_messages(prompt, system)

        base_url = self._config.ollama_base_url.rstrip("/")
        resp = await client.post(
            f"{base_url}/api/chat",
            json={
                "model": self._config.ollama_model,
                "messages": messages,
                "stream": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        content: str = data["message"]["content"]
        return content

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._config.timeout)
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _build_messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUSES
    return False
