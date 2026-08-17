"""Versioned public and artifact models."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChessInput(FrozenModel):
    fen: str
    variant: Literal["standard"]


class SourceReference(FrozenModel):
    move: str = Field(min_length=1)
    source_line: tuple[str, ...] | None = None


class NormalizedReference(FrozenModel):
    uci: str
    source_notation: str
    source_line: tuple[str, ...] | None = None


class SourceExample(FrozenModel):
    schema_version: Literal[1]
    id: str = Field(min_length=1)
    input: ChessInput
    reference: SourceReference | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("id must not be blank")
        return normalized


class NormalizedExample(FrozenModel):
    schema_version: Literal[1] = 1
    id: str
    input: ChessInput
    reference: NormalizedReference | None
    metadata: dict[str, JsonValue]


class ChatMessage(FrozenModel):
    role: Literal["system", "user", "assistant"]
    content: str


class GenerationRequest(FrozenModel):
    request_id: str
    example_id: str
    messages: tuple[ChatMessage, ...]
    parameters: dict[str, JsonValue]


class Usage(FrozenModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class GenerationSuccess(FrozenModel):
    outcome: Literal["success"] = "success"
    request_id: str
    example_id: str
    text: str
    finish_reason: str | None = None
    response_id: str | None = None
    response_model: str | None = None
    usage: Usage | None = None
    latency_ms: float = Field(ge=0)
    attempts: int = Field(default=1, ge=1)


class GenerationError(FrozenModel):
    outcome: Literal["error"] = "error"
    request_id: str
    example_id: str
    code: str
    transient: bool
    attempts: int = Field(ge=1)
    message: str
    latency_ms: float = Field(ge=0)
    error_id: str | None = None


GenerationOutcome = Annotated[
    GenerationSuccess | GenerationError,
    Field(discriminator="outcome"),
]


class MoveInterpretation(FrozenModel):
    parse_success: bool
    legal: bool
    predicted_uci: str | None
    notation: Literal["uci", "san"] | None
    candidate: str | None
    diagnostic: str
    format_compliant: bool
    extra_text: bool
    forced_move: bool


MetricStatus = Literal["measured", "skipped", "error"]


class MetricResult(FrozenModel):
    name: str
    version: str = "1"
    status: MetricStatus
    value: bool | int | float | str | None = None
    reason: str | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status_value(self) -> MetricResult:
        if self.status == "measured" and self.value is None:
            raise ValueError("measured metrics require a value")
        if self.status != "measured" and self.value is not None:
            raise ValueError("skipped and error metrics must not carry a value")
        return self


class EngineScore(FrozenModel):
    kind: Literal["cp", "mate"]
    value: int


class EngineAssessment(FrozenModel):
    forced_move: bool
    best_move_uci: str
    predicted_move_uci: str
    best_score: EngineScore
    predicted_score: EngineScore
    negative_loss_clamped: bool = False
    best_nodes: int | None = None
    predicted_nodes: int | None = None
    best_depth: int | None = None
    predicted_depth: int | None = None
    latency_ms: float = Field(default=0, ge=0)


class ValidationIssue(FrozenModel):
    code: str
    message: str
    line: int | None = None


class ValidationReport(FrozenModel):
    valid: bool
    example_count: int
    issue_count: int
    issues: tuple[ValidationIssue, ...]
    raw_sha256: str | None = None
    normalized_sha256: str | None = None
    engine_identity: dict[str, JsonValue] | None = None


RunStatus = Literal[
    "validation_failed",
    "failed",
    "complete",
    "partial",
    "interrupted",
]


class RunReport(FrozenModel):
    run_id: str
    status: RunStatus
    output_dir: Path
    total_examples: int
    processed_examples: int
    operational_errors: int

    @property
    def exit_code(self) -> int:
        return {
            "complete": 0,
            "partial": 2,
            "interrupted": 130,
            "validation_failed": 1,
            "failed": 1,
        }[self.status]


class ErrorRecord(FrozenModel):
    schema_version: Literal[1] = 1
    error_id: str
    code: str
    category: Literal["validation", "provider", "engine", "run", "artifact"]
    stage: str
    message: str
    line: int | None = None
    example_id: str | None = None
    request_id: str | None = None
    transient: bool | None = None
    attempts: int | None = None
