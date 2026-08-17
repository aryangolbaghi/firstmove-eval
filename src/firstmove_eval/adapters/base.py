"""Internal model adapter contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from firstmove_eval.models import GenerationOutcome, GenerationRequest


class ModelAdapter(Protocol):
    async def generate_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> list[GenerationOutcome]: ...

    async def aclose(self) -> None: ...
