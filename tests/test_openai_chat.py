from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from firstmove_eval.adapters.openai_chat import OpenAIChatAdapter
from firstmove_eval.config import OpenAIChatSettings
from firstmove_eval.models import ChatMessage, GenerationError, GenerationRequest, GenerationSuccess


def request(example_id: str = "one") -> GenerationRequest:
    return GenerationRequest(
        request_id=f"request-{example_id}",
        example_id=example_id,
        messages=(ChatMessage(role="user", content=example_id),),
        parameters={"temperature": 0, "max_tokens": 8},
    )


async def adapter_with_handler(handler: Any, *, max_attempts: int = 1) -> OpenAIChatAdapter:
    adapter = OpenAIChatAdapter(
        OpenAIChatSettings(
            base_url="https://provider.test/v1",
            model="model",
            max_attempts=max_attempts,
        )
    )
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="https://provider.test/v1/",
        transport=httpx.MockTransport(handler),
    )
    return adapter


@pytest.mark.asyncio
async def test_success_parses_only_one_text_choice_and_usage() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "response",
                "model": "served-model",
                "choices": [{"message": {"content": "e2e4"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    adapter = await adapter_with_handler(handler)
    try:
        result = (await adapter.generate_batch([request()]))[0]
    finally:
        await adapter.aclose()
    assert isinstance(result, GenerationSuccess)
    assert result.text == "e2e4"
    assert result.usage is not None and result.usage.total_tokens == 12


@pytest.mark.asyncio
async def test_auth_failure_is_not_retried_or_leaked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROVIDER_SECRET", "do-not-leak")
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="do-not-leak and provider internals")

    settings = OpenAIChatSettings(
        base_url="https://provider.test/v1",
        model="model",
        api_key_env="PROVIDER_SECRET",
        max_attempts=3,
    )
    adapter = OpenAIChatAdapter(settings)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="https://provider.test/v1/", transport=httpx.MockTransport(handler)
    )
    try:
        result = (await adapter.generate_batch([request()]))[0]
    finally:
        await adapter.aclose()
    assert isinstance(result, GenerationError)
    assert result.code == "provider_auth"
    assert calls == 1
    assert "do-not-leak" not in result.message


@pytest.mark.asyncio
async def test_rate_limit_retries_and_preserves_batch_order() -> None:
    calls: dict[str, int] = {}

    async def handler(http_request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(http_request.content)
        example_id = payload["messages"][0]["content"]
        calls[example_id] = calls.get(example_id, 0) + 1
        if example_id == "one" and calls[example_id] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        if example_id == "one":
            await asyncio.sleep(0.01)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": example_id}}]},
        )

    adapter = await adapter_with_handler(handler, max_attempts=2)
    try:
        results = await adapter.generate_batch([request("one"), request("two")])
    finally:
        await adapter.aclose()
    assert [result.example_id for result in results] == ["one", "two"]
    assert isinstance(results[0], GenerationSuccess)
    assert results[0].attempts == 2


@pytest.mark.asyncio
async def test_malformed_payload_becomes_terminal_error() -> None:
    adapter = await adapter_with_handler(lambda _: httpx.Response(200, json={"choices": []}))
    try:
        result = (await adapter.generate_batch([request()]))[0]
    finally:
        await adapter.aclose()
    assert isinstance(result, GenerationError)
    assert result.code == "provider_malformed_response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler", "expected_code"),
    [
        (
            lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("timeout", request=request)),
            "provider_timeout",
        ),
        (lambda _: httpx.Response(503), "provider_server_error"),
        (
            lambda _: httpx.Response(
                200,
                json={"choices": [{"message": {"content": None}}]},
            ),
            "provider_malformed_response",
        ),
    ],
)
async def test_terminal_provider_failures_have_stable_codes(
    handler: Any, expected_code: str
) -> None:
    adapter = await adapter_with_handler(handler)
    try:
        result = (await adapter.generate_batch([request()]))[0]
    finally:
        await adapter.aclose()
    assert isinstance(result, GenerationError)
    assert result.code == expected_code
