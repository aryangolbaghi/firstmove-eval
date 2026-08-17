"""Deterministic offline model adapter."""

from __future__ import annotations

from collections.abc import Sequence

from firstmove_eval.config import MockModelSettings
from firstmove_eval.models import GenerationOutcome, GenerationRequest, GenerationSuccess


class MockModelAdapter:
    def __init__(self, settings: MockModelSettings) -> None:
        self.settings = settings

    async def generate_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> list[GenerationOutcome]:
        return [
            GenerationSuccess(
                request_id=request.request_id,
                example_id=request.example_id,
                text=self.settings.responses.get(
                    request.example_id, self.settings.default_response
                ),
                finish_reason="stop",
                response_id=f"mock-{request.request_id}",
                response_model=self.settings.model,
                usage=None,
                latency_ms=0.0,
                attempts=1,
            )
            for request in requests
        ]

    async def aclose(self) -> None:
        return None
