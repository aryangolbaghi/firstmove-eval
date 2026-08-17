"""Strict, versioned configuration and safe manifest projection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

from firstmove_eval.errors import ConfigurationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunSettings(StrictModel):
    output_dir: Path = Path("runs")
    seed: int = 0


class DatasetSettings(StrictModel):
    path: Path


class TaskSettings(StrictModel):
    kind: Literal["chess_first_move"] = "chess_first_move"
    prompt_version: Literal["chess-uci-v1"] = "chess-uci-v1"


class GenerationParameters(StrictModel):
    temperature: float | None = Field(default=0, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: PositiveInt = 32
    seed: int | None = None
    stop: str | tuple[str, ...] | None = None

    def request_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude_none=True)


class MockModelSettings(StrictModel):
    kind: Literal["mock"] = "mock"
    model: str = "deterministic-mock-v1"
    batch_size: PositiveInt = 16
    responses: dict[str, str] = Field(default_factory=dict)
    default_response: str = ""
    parameters: GenerationParameters = Field(default_factory=GenerationParameters)


class OpenAIChatSettings(StrictModel):
    kind: Literal["openai_chat"] = "openai_chat"
    base_url: str
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    model: str = Field(min_length=1)
    batch_size: PositiveInt = 16
    max_concurrency: PositiveInt = 16
    timeout_seconds: PositiveFloat = 60
    max_attempts: PositiveInt = 3
    parameters: GenerationParameters = Field(default_factory=GenerationParameters)

    @model_validator(mode="after")
    def validate_endpoint(self) -> OpenAIChatSettings:
        normalized = self.base_url.rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("base_url must use http:// or https://")
        if not normalized.endswith("/v1"):
            raise ValueError("base_url must include and end with /v1")
        parsed = urlsplit(normalized)
        if not parsed.hostname:
            raise ValueError("base_url must include a hostname")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "base_url must not contain credentials, query parameters, or a fragment"
            )
        object.__setattr__(self, "base_url", normalized)
        return self


ModelSettings = Annotated[
    MockModelSettings | OpenAIChatSettings,
    Field(discriminator="kind"),
]


class EngineSettings(StrictModel):
    enabled: bool = False
    path: Path | None = None
    nodes: PositiveInt = 100_000
    threads: Literal[1] = 1
    hash_mb: PositiveInt = 128
    timeout_seconds: PositiveFloat = 60

    @model_validator(mode="after")
    def require_path_when_enabled(self) -> EngineSettings:
        if self.enabled and self.path is None:
            raise ValueError("engine.path is required when engine metrics are enabled")
        return self


class MetricsSettings(StrictModel):
    reference: bool = True
    engine: EngineSettings = Field(default_factory=EngineSettings)


class ArtifactSettings(StrictModel):
    store_prompts: bool = True
    store_raw_responses: bool = True


class RunConfig(StrictModel):
    schema_version: Literal[1] = 1
    run: RunSettings = Field(default_factory=RunSettings)
    dataset: DatasetSettings
    task: TaskSettings = Field(default_factory=TaskSettings)
    model: ModelSettings
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    artifacts: ArtifactSettings = Field(default_factory=ArtifactSettings)

    def safe_manifest_dict(self) -> dict[str, object]:
        model: dict[str, object]
        if isinstance(self.model, MockModelSettings):
            model = {
                "kind": self.model.kind,
                "model": self.model.model,
                "batch_size": self.model.batch_size,
                "response_count": len(self.model.responses),
                "has_default_response": bool(self.model.default_response),
                "parameters": self.model.parameters.request_dict(),
            }
        else:
            model = {
                "kind": self.model.kind,
                "base_url": self.model.base_url,
                "api_key_env": self.model.api_key_env,
                "model": self.model.model,
                "batch_size": self.model.batch_size,
                "max_concurrency": self.model.max_concurrency,
                "timeout_seconds": self.model.timeout_seconds,
                "max_attempts": self.model.max_attempts,
                "parameters": self.model.parameters.request_dict(),
            }
        return {
            "schema_version": self.schema_version,
            "run": {
                "output_dir": str(self.run.output_dir),
                "seed": self.run.seed,
            },
            "dataset": {"path": str(self.dataset.path)},
            "task": self.task.model_dump(mode="json"),
            "model": model,
            "metrics": self.metrics.model_dump(mode="json"),
            "artifacts": self.artifacts.model_dump(mode="json"),
        }


def _resolve_paths(config: RunConfig, base_dir: Path) -> RunConfig:
    dataset_path = config.dataset.path
    output_dir = config.run.output_dir
    if not dataset_path.is_absolute():
        dataset_path = (base_dir / dataset_path).resolve()
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()

    engine = config.metrics.engine
    engine_path = engine.path
    if engine_path is not None and not engine_path.is_absolute():
        engine_path = (base_dir / engine_path).resolve()

    return config.model_copy(
        update={
            "run": config.run.model_copy(update={"output_dir": output_dir}),
            "dataset": config.dataset.model_copy(update={"path": dataset_path}),
            "metrics": config.metrics.model_copy(
                update={"engine": engine.model_copy(update={"path": engine_path})}
            ),
        }
    )


def load_config(path: str | Path) -> RunConfig:
    """Load a YAML config, forbid unknown fields, and resolve relative paths."""

    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read configuration {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("configuration root must be a YAML mapping")
    try:
        config = RunConfig.model_validate(raw)
    except ValueError as exc:
        raise ConfigurationError(f"invalid configuration: {exc}") from exc
    return _resolve_paths(config, config_path.parent)


def validate_runtime_configuration(config: RunConfig) -> list[tuple[str, str]]:
    """Return safe startup problems that do not require a model request."""

    issues: list[tuple[str, str]] = []
    if not config.dataset.path.is_file():
        issues.append(("dataset_not_found", f"dataset does not exist: {config.dataset.path}"))
    if (
        isinstance(config.model, OpenAIChatSettings)
        and config.model.api_key_env
        and not os.environ.get(config.model.api_key_env)
    ):
        issues.append(
            (
                "credential_missing",
                f"credential environment variable is not set: {config.model.api_key_env}",
            )
        )
    engine = config.metrics.engine
    if engine.enabled and (engine.path is None or not engine.path.is_file()):
        issues.append(("engine_not_found", f"Stockfish executable does not exist: {engine.path}"))
    return issues
