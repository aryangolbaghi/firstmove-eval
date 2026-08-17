"""Minimal OpenAI-compatible Chat Completions adapter."""

from __future__ import annotations

import asyncio
import email.utils
import os
import random
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import httpx

from firstmove_eval.config import OpenAIChatSettings
from firstmove_eval.models import (
    GenerationError,
    GenerationOutcome,
    GenerationRequest,
    GenerationSuccess,
    Usage,
)


class OpenAIChatAdapter:
    def __init__(self, settings: OpenAIChatSettings, *, seed: int = 0) -> None:
        self.settings = settings
        headers = {"Content-Type": "application/json"}
        if settings.api_key_env:
            api_key = os.environ.get(settings.api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=f"{settings.base_url}/",
            headers=headers,
            timeout=float(settings.timeout_seconds),
        )
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        self._random = random.Random(seed)

    async def generate_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> list[GenerationOutcome]:
        tasks = [asyncio.create_task(self._generate_one(request)) for request in requests]
        return list(await asyncio.gather(*tasks))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _generate_one(self, request: GenerationRequest) -> GenerationOutcome:
        started = time.perf_counter()
        async with self._semaphore:
            for attempt in range(1, self.settings.max_attempts + 1):
                try:
                    response = await self._client.post(
                        "chat/completions",
                        json={
                            "model": self.settings.model,
                            "messages": [
                                message.model_dump(mode="json") for message in request.messages
                            ],
                            "n": 1,
                            **request.parameters,
                        },
                    )
                except asyncio.CancelledError:
                    raise
                except httpx.TimeoutException:
                    if attempt < self.settings.max_attempts:
                        await self._backoff(attempt, None)
                        continue
                    return self._error(
                        request,
                        "provider_timeout",
                        True,
                        attempt,
                        "provider request timed out",
                        started,
                    )
                except httpx.TransportError:
                    if attempt < self.settings.max_attempts:
                        await self._backoff(attempt, None)
                        continue
                    return self._error(
                        request,
                        "provider_transport",
                        True,
                        attempt,
                        "provider transport failed",
                        started,
                    )

                if response.status_code in {408, 429} or response.status_code >= 500:
                    if attempt < self.settings.max_attempts:
                        await self._backoff(attempt, response.headers.get("Retry-After"))
                        continue
                    code = (
                        "provider_rate_limit"
                        if response.status_code == 429
                        else (
                            "provider_timeout"
                            if response.status_code == 408
                            else "provider_server_error"
                        )
                    )
                    return self._error(
                        request,
                        code,
                        True,
                        attempt,
                        f"provider returned HTTP {response.status_code}",
                        started,
                    )
                if response.status_code in {401, 403}:
                    return self._error(
                        request,
                        "provider_auth",
                        False,
                        attempt,
                        f"provider rejected credentials with HTTP {response.status_code}",
                        started,
                    )
                if response.status_code >= 400:
                    return self._error(
                        request,
                        "provider_request_rejected",
                        False,
                        attempt,
                        f"provider rejected request with HTTP {response.status_code}",
                        started,
                    )
                try:
                    payload: Any = response.json()
                    return self._parse_success(request, payload, attempt, started)
                except (ValueError, KeyError, TypeError, IndexError):
                    return self._error(
                        request,
                        "provider_malformed_response",
                        False,
                        attempt,
                        "provider returned a malformed Chat Completions response",
                        started,
                    )
        raise AssertionError("unreachable retry loop")

    def _parse_success(
        self,
        request: GenerationRequest,
        payload: Any,
        attempts: int,
        started: float,
    ) -> GenerationSuccess:
        if not isinstance(payload, dict):
            raise TypeError("payload is not an object")
        choices = payload["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("expected one choice")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise TypeError("choice is not an object")
        message = choice["message"]
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise TypeError("choice content is not text")
        usage_payload = payload.get("usage")
        usage = None
        if isinstance(usage_payload, dict):
            usage = Usage(
                prompt_tokens=_optional_int(usage_payload.get("prompt_tokens")),
                completion_tokens=_optional_int(usage_payload.get("completion_tokens")),
                total_tokens=_optional_int(usage_payload.get("total_tokens")),
            )
        finish_reason = choice.get("finish_reason")
        return GenerationSuccess(
            request_id=request.request_id,
            example_id=request.example_id,
            text=message["content"],
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            response_id=payload.get("id") if isinstance(payload.get("id"), str) else None,
            response_model=(
                payload.get("model") if isinstance(payload.get("model"), str) else None
            ),
            usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000,
            attempts=attempts,
        )

    def _error(
        self,
        request: GenerationRequest,
        code: str,
        transient: bool,
        attempts: int,
        message: str,
        started: float,
    ) -> GenerationError:
        return GenerationError(
            request_id=request.request_id,
            example_id=request.example_id,
            code=code,
            transient=transient,
            attempts=attempts,
            message=message,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        seconds = _retry_after_seconds(retry_after)
        if seconds is None:
            seconds = min(30.0, 0.5 * (2 ** (attempt - 1)))
            seconds += self._random.uniform(0, seconds * 0.25)
        seconds = min(30.0, seconds)
        await asyncio.sleep(seconds)


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            target = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=UTC)
        return max(0.0, (target - datetime.now(UTC)).total_seconds())
