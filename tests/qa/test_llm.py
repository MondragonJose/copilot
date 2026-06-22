"""Tests for the LLM provider — all HTTP calls are mocked."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from qa.llm import (
    LLMConfig,
    LLMError,
    LLMProvider,
    _build_messages,
    _is_retryable,
)


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------

class TestLLMConfig:
    def test_default_values(self) -> None:
        cfg = LLMConfig()
        assert cfg.provider == "openai"
        assert cfg.api_key is None
        assert cfg.model == "gpt-4o-mini"
        assert cfg.timeout == 30.0
        assert cfg.max_retries == 3

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_MODEL", "llama3")
        cfg = LLMConfig.from_env()
        assert cfg.provider == "ollama"
        assert cfg.api_key == "sk-test"
        assert cfg.model == "llama3"

    def test_from_env_populates_all_fields(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
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


# ---------------------------------------------------------------------------
# LLMProvider
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client() -> httpx.AsyncClient:
    client = MagicMock(spec=httpx.AsyncClient)
    client.post = AsyncMock()
    return client


class TestLLMProvider:
    @pytest.mark.asyncio
    async def test_openai_success(self, mock_client: httpx.AsyncClient) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json = MagicMock(
            return_value={"choices": [{"message": {"content": "Hello!"}}]},
        )
        mock_client.post.return_value = mock_response

        cfg = LLMConfig(provider="openai", api_key="sk-test", base_url=None)
        provider = LLMProvider(config=cfg, client=mock_client)
        result = await provider.generate("Hi", system="Be nice.")
        assert result == "Hello!"
        assert mock_client.post.call_count >= 1

    @pytest.mark.asyncio
    async def test_ollama_success(self, mock_client: httpx.AsyncClient) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json = MagicMock(
            return_value={"message": {"content": "Ollama answer"}},
        )
        mock_client.post.return_value = mock_response

        cfg = LLMConfig(provider="ollama", ollama_base_url="http://ollama:11434",
                        ollama_model="llama3")
        provider = LLMProvider(config=cfg, client=mock_client)
        result = await provider.generate("Hi")
        assert result == "Ollama answer"

    @pytest.mark.asyncio
    async def test_retry_then_success(self, mock_client: httpx.AsyncClient) -> None:
        fail_response = MagicMock(spec=httpx.Response)
        fail_response.status_code = 429
        fail_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "too many", request=MagicMock(), response=fail_response,
        )

        ok_response = MagicMock(spec=httpx.Response)
        ok_response.status_code = 200
        ok_response.json = MagicMock(
            return_value={"choices": [{"message": {"content": "OK"}}]},
        )

        mock_client.post.side_effect = [fail_response, ok_response]
        cfg = LLMConfig(provider="openai", api_key="sk-test",
                        max_retries=3, timeout=5.0)
        provider = LLMProvider(config=cfg, client=mock_client)
        result = await provider.generate("Hi")
        assert result == "OK"
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self, mock_client: httpx.AsyncClient) -> None:
        fail_response = MagicMock(spec=httpx.Response)
        fail_response.status_code = 503
        fail_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "down", request=MagicMock(), response=fail_response,
        )
        mock_client.post.side_effect = [fail_response] * 3

        cfg = LLMConfig(provider="openai", api_key="sk-test",
                        max_retries=3, timeout=5.0)
        provider = LLMProvider(config=cfg, client=mock_client)
        with pytest.raises(LLMError, match="down"):
            await provider.generate("Hi")

    @pytest.mark.asyncio
    async def test_non_retryable_status_raises_immediately(
        self, mock_client: httpx.AsyncClient,
    ) -> None:
        fail_response = MagicMock(spec=httpx.Response)
        fail_response.status_code = 400
        fail_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "bad request", request=MagicMock(), response=fail_response,
        )
        mock_client.post.side_effect = [fail_response] * 5  # all calls fail

        cfg = LLMConfig(provider="openai", api_key="sk-test", max_retries=3)
        provider = LLMProvider(config=cfg, client=mock_client)
        with pytest.raises(LLMError, match="bad request"):
            await provider.generate("Hi")
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_timeout_retries(self, mock_client: httpx.AsyncClient) -> None:
        mock_client.post.side_effect = [
            httpx.TimeoutException("timeout", request=MagicMock()),
            httpx.TimeoutException("timeout", request=MagicMock()),
        ]
        cfg = LLMConfig(provider="openai", api_key="sk-test", max_retries=2)
        provider = LLMProvider(config=cfg, client=mock_client)
        with pytest.raises(LLMError, match="timeout"):
            await provider.generate("Hi")

    @pytest.mark.asyncio
    async def test_aclose_owned_client(self) -> None:
        """When no client is injected, aclose should succeed gracefully."""
        cfg = LLMConfig(provider="openai", api_key="sk-test")
        provider = LLMProvider(config=cfg)
        # No client yet — aclose should not crash
        await provider.aclose()

    @pytest.mark.asyncio
    async def test_system_prompt_omitted(self, mock_client: httpx.AsyncClient) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json = MagicMock(
            return_value={"choices": [{"message": {"content": "A"}}]},
        )
        mock_client.post.return_value = mock_response
        cfg = LLMConfig(provider="openai", api_key="sk-test",
                        base_url="https://fake.api.com/v1")
        provider = LLMProvider(config=cfg, client=mock_client)
        result = await provider.generate("Hi")
        assert result == "A"

    @pytest.mark.asyncio
    async def test_custom_base_url(self, mock_client: httpx.AsyncClient) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json = MagicMock(
            return_value={"choices": [{"message": {"content": "A"}}]},
        )
        mock_client.post.return_value = mock_response
        cfg = LLMConfig(provider="openai", api_key="sk-test",
                        base_url="https://custom.api.com")
        provider = LLMProvider(config=cfg, client=mock_client)
        await provider.generate("Hi")
        call_url = str(mock_client.post.call_args[0][0])
        assert "custom.api.com" in call_url

    @pytest.mark.asyncio
    async def test_sends_authorization_header(
        self, mock_client: httpx.AsyncClient,
    ) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json = MagicMock(
            return_value={"choices": [{"message": {"content": "OK"}}]},
        )
        mock_client.post.return_value = mock_response
        cfg = LLMConfig(provider="openai", api_key="sk-custom-key")
        provider = LLMProvider(config=cfg, client=mock_client)
        await provider.generate("Hi")
        call_kwargs = mock_client.post.call_args[1]
        assert "Authorization" in call_kwargs.get("headers", {})
        assert call_kwargs["headers"]["Authorization"] == "Bearer sk-custom-key"

    @pytest.mark.asyncio
    async def test_no_api_key_skips_auth_header(
        self, mock_client: httpx.AsyncClient,
    ) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json = MagicMock(
            return_value={"choices": [{"message": {"content": "OK"}}]},
        )
        mock_client.post.return_value = mock_response
        cfg = LLMConfig(provider="openai", api_key=None)
        provider = LLMProvider(config=cfg, client=mock_client)
        await provider.generate("Hi")
        call_kwargs = mock_client.post.call_args[1]
        headers = call_kwargs.get("headers", {})
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_retries_on_timeout_then_succeeds(
        self, mock_client: httpx.AsyncClient,
    ) -> None:
        ok_response = MagicMock(spec=httpx.Response)
        ok_response.status_code = 200
        ok_response.json = MagicMock(
            return_value={"choices": [{"message": {"content": "After timeout"}}]},
        )
        mock_client.post.side_effect = [
            httpx.TimeoutException("timeout", request=MagicMock()),
            ok_response,
        ]
        cfg = LLMConfig(provider="openai", api_key="sk-test",
                        max_retries=2, timeout=5.0)
        provider = LLMProvider(config=cfg, client=mock_client)
        result = await provider.generate("Hi")
        assert result == "After timeout"
        assert mock_client.post.call_count == 2


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestBuildMessages:
    def test_with_system(self) -> None:
        msgs = _build_messages("user prompt", "system prompt")
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "system prompt"}
        assert msgs[1] == {"role": "user", "content": "user prompt"}

    def test_without_system(self) -> None:
        msgs = _build_messages("user prompt", None)
        assert len(msgs) == 1
        assert msgs[0] == {"role": "user", "content": "user prompt"}

    def test_empty_prompt(self) -> None:
        msgs = _build_messages("", "sys")
        assert msgs[1]["content"] == ""


class TestIsRetryable:
    def test_timeout_is_retryable(self) -> None:
        exc = httpx.TimeoutException("timeout", request=MagicMock())
        assert _is_retryable(exc) is True

    def test_429_is_retryable(self) -> None:
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 429
        exc = httpx.HTTPStatusError("429", request=MagicMock(), response=resp)
        assert _is_retryable(exc) is True

    def test_500_is_retryable(self) -> None:
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 500
        exc = httpx.HTTPStatusError("500", request=MagicMock(), response=resp)
        assert _is_retryable(exc) is True

    def test_400_is_not_retryable(self) -> None:
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 400
        exc = httpx.HTTPStatusError("400", request=MagicMock(), response=resp)
        assert _is_retryable(exc) is False

    def test_unknown_exc_not_retryable(self) -> None:
        assert _is_retryable(ValueError("nope")) is False


# ---------------------------------------------------------------------------
# Provider swap
# ---------------------------------------------------------------------------


class TestProviderSwap:
    @pytest.mark.asyncio
    async def test_swap_from_openai_to_ollama(self) -> None:
        openai_cfg = LLMConfig(provider="openai", api_key="sk-test")
        ollama_cfg = LLMConfig(provider="ollama")

        assert openai_cfg.provider == "openai"
        assert ollama_cfg.provider == "ollama"
