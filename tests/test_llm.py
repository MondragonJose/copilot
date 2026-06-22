"""Adapter tests for LLMProvider — mocked HTTP transport."""

from __future__ import annotations

import json

import httpx
import pytest

from qa.llm import LLMConfig, LLMError, LLMProvider

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------
# Helpers — mock transport handlers
# --------------------------------------------------------------------------


def _ok_response(text: str = "Hello from LLM") -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": text}}]},
    )


def _ollama_response(text: str = "Hello from Ollama") -> httpx.Response:
    return httpx.Response(
        200,
        json={"message": {"content": text}},
    )


def _error_response(status: int = 429) -> httpx.Response:
    return httpx.Response(status)


# --------------------------------------------------------------------------
# OpenAI-compatible provider
# --------------------------------------------------------------------------


class TestOpenAIProvider:

    async def test_generate_returns_text(self) -> None:
        async def handler(_: httpx.Request) -> httpx.Response:
            return _ok_response("Hello world")

        transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = LLMProvider(client=transport)

        result = await provider.generate("Hi")

        assert result == "Hello world"

    async def test_generate_with_system_prompt(self) -> None:
        async def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            messages = body["messages"]
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "Be helpful"
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "Question?"
            return _ok_response("Answer")

        transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = LLMProvider(client=transport)

        result = await provider.generate("Question?", system="Be helpful")
        assert result == "Answer"

    async def test_sends_authorization_header(self) -> None:
        async def handler(req: httpx.Request) -> httpx.Response:
            assert req.headers["Authorization"] == "Bearer sk-custom-key"
            return _ok_response("OK")

        transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        cfg = LLMConfig(api_key="sk-custom-key")
        provider = LLMProvider(config=cfg, client=transport)

        await provider.generate("Hi")

    async def test_uses_custom_base_url(self) -> None:
        async def handler(req: httpx.Request) -> httpx.Response:
            assert str(req.url).startswith("https://custom.example.com/v1/")
            return _ok_response("OK")

        transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        cfg = LLMConfig(base_url="https://custom.example.com/v1")
        provider = LLMProvider(config=cfg, client=transport)

        await provider.generate("Hi")

    async def test_no_api_key_skips_auth_header(self) -> None:
        async def handler(req: httpx.Request) -> httpx.Response:
            assert "Authorization" not in req.headers
            return _ok_response("OK")

        transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        cfg = LLMConfig(api_key=None)
        provider = LLMProvider(config=cfg, client=transport)

        await provider.generate("Hi")


# --------------------------------------------------------------------------
# Ollama provider
# --------------------------------------------------------------------------


class TestOllamaProvider:

    async def test_generate_returns_text(self) -> None:
        async def handler(_: httpx.Request) -> httpx.Response:
            return _ollama_response("Ollama answer")

        transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        cfg = LLMConfig(provider="ollama")
        provider = LLMProvider(config=cfg, client=transport)

        result = await provider.generate("Hi")

        assert result == "Ollama answer"

    async def test_sends_correct_payload(self) -> None:
        async def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            assert body["model"] == "llama3.2"
            assert not body["stream"]
            assert body["messages"][0]["role"] == "user"
            assert body["messages"][0]["content"] == "Hello"
            return _ollama_response("Hi")

        transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        cfg = LLMConfig(provider="ollama")
        provider = LLMProvider(config=cfg, client=transport)

        await provider.generate("Hello")

    async def test_uses_custom_ollama_base_url(self) -> None:
        async def handler(req: httpx.Request) -> httpx.Response:
            assert str(req.url).startswith("http://my-ollama:11434/api/")
            return _ollama_response("OK")

        transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        cfg = LLMConfig(provider="ollama", ollama_base_url="http://my-ollama:11434")
        provider = LLMProvider(config=cfg, client=transport)

        await provider.generate("Hi")

    async def test_ollama_disabled_by_default(self) -> None:
        cfg = LLMConfig()
        assert cfg.provider == "openai"


# --------------------------------------------------------------------------
# Retry
# --------------------------------------------------------------------------


class TestRetry:

    async def test_retries_on_429_then_succeeds(self) -> None:
        call_count = 0

        async def handler(_: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return _error_response(429)
            return _ok_response("Success after retry")

        transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        cfg = LLMConfig(max_retries=3, timeout=5.0)
        provider = LLMProvider(config=cfg, client=transport)

        result = await provider.generate("Hi")

        assert result == "Success after retry"
        assert call_count == 3

    async def test_retries_on_timeout_then_succeeds(self) -> None:
        call_count = 0

        async def handler(_: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                msg = "simulated timeout"
                raise httpx.TimeoutException(msg)
            return _ok_response("After timeout")

        transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        cfg = LLMConfig(max_retries=2, timeout=5.0)
        provider = LLMProvider(config=cfg, client=transport)

        result = await provider.generate("Hi")

        assert result == "After timeout"
        assert call_count == 2

    async def test_exhausted_retries_raises_llm_error(self) -> None:
        async def handler(_: httpx.Request) -> httpx.Response:
            return _error_response(503)

        transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        cfg = LLMConfig(max_retries=2, timeout=5.0)
        provider = LLMProvider(config=cfg, client=transport)

        with pytest.raises(LLMError):
            await provider.generate("Hi")

    async def test_non_retryable_status_raises_immediately(self) -> None:
        async def handler(_: httpx.Request) -> httpx.Response:
            return _error_response(400)

        transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        cfg = LLMConfig(max_retries=3, timeout=5.0)
        provider = LLMProvider(config=cfg, client=transport)

        with pytest.raises(LLMError):
            await provider.generate("Hi")


# --------------------------------------------------------------------------
# Config from env
# --------------------------------------------------------------------------


class TestLLMConfigFromEnv:

    def test_defaults(self) -> None:
        cfg = LLMConfig()
        assert cfg.provider == "openai"
        assert cfg.api_key is None
        assert cfg.base_url is None
        assert cfg.model == "gpt-4o-mini"
        assert cfg.timeout == 30.0
        assert cfg.max_retries == 3
        assert cfg.ollama_base_url == "http://localhost:11434"
        assert cfg.ollama_model == "llama3.2"

    def test_from_env_populates_all_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_API_KEY", "env-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://env.example.com/v1")
        monkeypatch.setenv("LLM_MODEL", "gpt-4")
        monkeypatch.setenv("LLM_TIMEOUT", "60.0")
        monkeypatch.setenv("LLM_MAX_RETRIES", "5")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-ollama:11434")
        monkeypatch.setenv("OLLAMA_MODEL", "llama3")

        cfg = LLMConfig.from_env()

        assert cfg.provider == "ollama"
        assert cfg.api_key == "env-key"
        assert cfg.base_url == "https://env.example.com/v1"
        assert cfg.model == "gpt-4"
        assert cfg.timeout == 60.0
        assert cfg.max_retries == 5
        assert cfg.ollama_base_url == "http://env-ollama:11434"
        assert cfg.ollama_model == "llama3"


# --------------------------------------------------------------------------
# Provider swap via config
# --------------------------------------------------------------------------


class TestProviderSwap:

    async def test_swap_from_openai_to_ollama(self) -> None:
        openai_cfg = LLMConfig(provider="openai", api_key="sk-test")
        ollama_cfg = LLMConfig(provider="ollama")

        assert openai_cfg.provider == "openai"
        assert ollama_cfg.provider == "ollama"

        async def openai_handler(req: httpx.Request) -> httpx.Response:
            assert "/v1/chat/completions" in str(req.url)
            return _ok_response("OpenAI response")

        async def ollama_handler(req: httpx.Request) -> httpx.Response:
            assert "/api/chat" in str(req.url)
            return _ollama_response("Ollama response")

        openai_provider = LLMProvider(
            config=openai_cfg,
            client=httpx.AsyncClient(transport=httpx.MockTransport(openai_handler)),
        )
        ollama_provider = LLMProvider(
            config=ollama_cfg,
            client=httpx.AsyncClient(transport=httpx.MockTransport(ollama_handler)),
        )

        openai_result = await openai_provider.generate("Hi")
        ollama_result = await ollama_provider.generate("Hi")

        assert openai_result == "OpenAI response"
        assert ollama_result == "Ollama response"
