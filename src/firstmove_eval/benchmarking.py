"""Public benchmark plugin contract, discovery, and configuration."""

from __future__ import annotations

import importlib.metadata
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from firstmove_eval.errors import ConfigurationError

BENCHMARK_ENTRY_POINT_GROUP = "firstmove_eval.benchmarks"


class StrictBenchmarkModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BenchmarkJobConfig(StrictBenchmarkModel):
    """One named execution of a benchmark plugin."""

    plugin: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    name: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    work_items: PositiveInt
    repetitions: PositiveInt = 1
    options: dict[str, JsonValue] = Field(default_factory=dict)
    max_peak_rss_mib: PositiveFloat | None = None

    @property
    def result_name(self) -> str:
        return self.name or self.plugin


class BenchmarkRunSettings(StrictBenchmarkModel):
    output: Path | None = None
    baseline: Path | None = None
    max_slowdown_percent: float = Field(default=20.0, ge=0)


class BenchmarkSuiteConfig(StrictBenchmarkModel):
    schema_version: Literal[1] = 1
    name: str = Field(default="custom", min_length=1)
    run: BenchmarkRunSettings = Field(default_factory=BenchmarkRunSettings)
    benchmarks: tuple[BenchmarkJobConfig, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_result_names(self) -> BenchmarkSuiteConfig:
        names = [job.result_name for job in self.benchmarks]
        if len(names) != len(set(names)):
            raise ValueError("benchmark names must be unique within a suite")
        return self


@dataclass(frozen=True, slots=True)
class BenchmarkContext:
    """Inputs supplied to a benchmark plugin inside its worker process."""

    root: Path
    work_items: int
    repetitions: int
    options: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class BenchmarkMeasurement:
    """Raw elapsed-time samples returned by a benchmark plugin."""

    samples_seconds: tuple[float, ...]

    def __post_init__(self) -> None:
        samples = tuple(float(sample) for sample in self.samples_seconds)
        if not samples or any(sample <= 0 or not math.isfinite(sample) for sample in samples):
            raise ValueError("benchmark samples must be finite values greater than zero")
        object.__setattr__(self, "samples_seconds", samples)


class BenchmarkPlugin(Protocol):
    """Contract implemented by built-in and third-party benchmark plugins."""

    description: str

    def measure(self, context: BenchmarkContext) -> BenchmarkMeasurement: ...


@dataclass(frozen=True, slots=True)
class PluginDescriptor:
    name: str
    source: str
    description: str


class BenchmarkPluginError(RuntimeError):
    """A selected benchmark plugin is unavailable or violates the plugin contract."""


def load_benchmark_config(path: str | Path) -> BenchmarkSuiteConfig:
    """Load a strict YAML suite and resolve result paths relative to the config file."""

    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read benchmark config {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("benchmark configuration root must be a YAML mapping")
    try:
        config = BenchmarkSuiteConfig.model_validate(raw)
    except ValueError as exc:
        raise ConfigurationError(f"invalid benchmark configuration: {exc}") from exc

    base_dir = config_path.parent
    output = _resolve_optional_path(config.run.output, base_dir)
    baseline = _resolve_optional_path(config.run.baseline, base_dir)
    return config.model_copy(
        update={"run": config.run.model_copy(update={"output": output, "baseline": baseline})}
    )


def plugin_descriptors() -> tuple[PluginDescriptor, ...]:
    """List built-ins and installed entry points without importing third-party plugin code."""

    from firstmove_eval.benchmark_plugins import BUILTIN_BENCHMARK_PLUGINS

    descriptors = [
        PluginDescriptor(name=name, source="built-in", description=factory().description)
        for name, factory in BUILTIN_BENCHMARK_PLUGINS.items()
    ]
    for entry_point in _external_entry_points():
        distribution = entry_point.dist.name if entry_point.dist else "unknown distribution"
        descriptors.append(
            PluginDescriptor(
                name=entry_point.name,
                source=distribution,
                description=f"installed entry point {entry_point.value}",
            )
        )
    return tuple(sorted(descriptors, key=lambda item: (item.name, item.source)))


def load_benchmark_plugin(name: str) -> BenchmarkPlugin:
    """Load one explicitly selected built-in or installed entry-point plugin."""

    from firstmove_eval.benchmark_plugins import BUILTIN_BENCHMARK_PLUGINS

    builtin = BUILTIN_BENCHMARK_PLUGINS.get(name)
    if builtin is not None:
        return builtin()

    matches = [entry_point for entry_point in _external_entry_points() if entry_point.name == name]
    if not matches:
        available = ", ".join(descriptor.name for descriptor in plugin_descriptors())
        raise BenchmarkPluginError(
            f"benchmark plugin {name!r} is not installed; available plugins: {available}"
        )
    if len(matches) > 1:
        sources = ", ".join(
            entry_point.dist.name if entry_point.dist else entry_point.value
            for entry_point in matches
        )
        raise BenchmarkPluginError(f"benchmark plugin {name!r} is ambiguous: {sources}")

    try:
        loaded: Any = matches[0].load()
        candidate = _instantiate_plugin(loaded)
    except Exception as exc:
        raise BenchmarkPluginError(
            f"could not load benchmark plugin {name!r}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(getattr(candidate, "description", None), str) or not callable(
        getattr(candidate, "measure", None)
    ):
        raise BenchmarkPluginError(
            f"benchmark plugin {name!r} must provide a description and measure(context)"
        )
    return cast(BenchmarkPlugin, candidate)


def _instantiate_plugin(loaded: Any) -> Any:
    if isinstance(loaded, type):
        return loaded()
    if callable(loaded) and not callable(getattr(loaded, "measure", None)):
        return loaded()
    return loaded


def _external_entry_points() -> tuple[importlib.metadata.EntryPoint, ...]:
    return tuple(importlib.metadata.entry_points(group=BENCHMARK_ENTRY_POINT_GROUP))


def _resolve_optional_path(path: Path | None, base_dir: Path) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return (base_dir / path).resolve()


BenchmarkPluginFactory = Callable[[], BenchmarkPlugin]
