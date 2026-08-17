"""Stable internal exception and diagnostic types."""

from __future__ import annotations

from dataclasses import dataclass


class FirstMoveEvalError(Exception):
    """Base exception for fatal evaluation failures."""


class ConfigurationError(FirstMoveEvalError):
    """Configuration cannot be used safely."""


class ArtifactError(FirstMoveEvalError):
    """A required artifact could not be persisted."""


class EngineEvaluationError(FirstMoveEvalError):
    """Stockfish failed for one position after its bounded retry."""


@dataclass(frozen=True, slots=True)
class DatasetRowError(Exception):
    """A source row cannot be normalized into a chess example."""

    code: str
    message: str

    def __str__(self) -> str:
        return self.message
