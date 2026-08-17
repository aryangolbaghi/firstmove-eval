"""Public validation and run lifecycle."""

from __future__ import annotations

import asyncio
import importlib.metadata
import platform
import secrets
import subprocess
from collections.abc import Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from firstmove_eval._json import canonical_json, sha256_text
from firstmove_eval._version import __version__
from firstmove_eval.adapters.base import ModelAdapter
from firstmove_eval.adapters.mock import MockModelAdapter
from firstmove_eval.adapters.openai_chat import OpenAIChatAdapter
from firstmove_eval.artifacts import ArtifactStore
from firstmove_eval.config import (
    MockModelSettings,
    OpenAIChatSettings,
    RunConfig,
    validate_runtime_configuration,
)
from firstmove_eval.engine import ChessEngineService
from firstmove_eval.metrics import ChessMetricSuite, EvaluationBundle, SummaryAccumulator
from firstmove_eval.models import (
    ErrorRecord,
    GenerationError,
    GenerationOutcome,
    GenerationRequest,
    GenerationSuccess,
    MoveInterpretation,
    NormalizedExample,
    RunReport,
    RunStatus,
    ValidationIssue,
    ValidationReport,
)
from firstmove_eval.source import JsonlSource, PreflightResult
from firstmove_eval.task import ChessFirstMoveTask


def validate(config: RunConfig) -> ValidationReport:
    """Validate configuration, data, credentials, and engine startup without model calls."""

    issues = [
        ValidationIssue(code=code, message=message)
        for code, message in validate_runtime_configuration(config)
    ]
    task = ChessFirstMoveTask()
    preflight: PreflightResult | None = None
    if config.dataset.path.is_file():
        try:
            with TemporaryDirectory(prefix="firstmove-eval-validate-") as temporary:
                preflight = JsonlSource(config.dataset.path).preflight(
                    task, Path(temporary) / "input_snapshot.jsonl"
                )
        except OSError as exc:
            issues.append(
                ValidationIssue(
                    code="dataset_read_failed",
                    message=f"dataset preflight failed: {type(exc).__name__}",
                )
            )
        else:
            issues.extend(preflight.issues)

    engine_identity = None
    engine_settings = config.metrics.engine
    if engine_settings.enabled and engine_settings.path and engine_settings.path.is_file():
        engine = ChessEngineService(engine_settings)
        try:
            engine.start()
            engine_identity = engine.identity
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    code="engine_start_failed",
                    message=f"Stockfish startup failed: {type(exc).__name__}",
                )
            )
        finally:
            engine.close()

    runtime_issue_count = len(issues) - (len(preflight.issues) if preflight else 0)
    issue_count = runtime_issue_count + (preflight.issue_count if preflight else 0)
    return ValidationReport(
        valid=issue_count == 0,
        example_count=preflight.example_count if preflight else 0,
        issue_count=issue_count,
        issues=tuple(issues[:1_000]),
        raw_sha256=preflight.raw_sha256 if preflight else None,
        normalized_sha256=preflight.normalized_sha256 if preflight else None,
        engine_identity=engine_identity,
    )


def run(config: RunConfig) -> RunReport:
    """Run one blocking evaluation with an internal asynchronous provider pipeline."""

    return asyncio.run(_run_async(config))


async def _run_async(config: RunConfig) -> RunReport:
    run_id = _run_id()
    run_dir = config.run.output_dir / run_id
    store = ArtifactStore(run_dir)
    task = ChessFirstMoveTask()
    suite = ChessMetricSuite(config.metrics)
    accumulator = SummaryAccumulator(suite.metric_kinds)
    started_at = _timestamp()
    manifest = _initial_manifest(config, run_id, started_at)
    manifest["metric_definitions"] = suite.metric_definitions
    store.write_manifest(manifest)
    total_examples = 0

    runtime_issues = validate_runtime_configuration(config)
    if runtime_issues:
        for code, message in runtime_issues:
            store.write_error(_error_record(code, "run", "startup", message))
            accumulator.operational_errors += 1
        store.flush_batch()
        report = _finalize(
            store,
            manifest,
            accumulator,
            "failed",
            total_examples,
            run_id,
            run_dir,
        )
        store.close()
        return report

    def write_validation_issue(issue: ValidationIssue) -> None:
        store.write_error(
            _error_record(
                issue.code,
                "validation",
                "preflight",
                issue.message,
                line=issue.line,
            )
        )

    try:
        try:
            preflight = JsonlSource(config.dataset.path).preflight(
                task, store.snapshot_path, on_issue=write_validation_issue
            )
        except OSError as exc:
            store.write_error(
                _error_record(
                    "dataset_read_failed",
                    "run",
                    "preflight",
                    f"dataset preflight failed: {type(exc).__name__}",
                )
            )
            accumulator.operational_errors += 1
            store.flush_batch()
            return _finalize(
                store,
                manifest,
                accumulator,
                "failed",
                total_examples,
                run_id,
                run_dir,
            )
        total_examples = preflight.example_count
        manifest["dataset"] = {
            "path": str(config.dataset.path),
            "raw_sha256": preflight.raw_sha256,
            "normalized_sha256": preflight.normalized_sha256,
            "example_count": preflight.example_count,
        }
        if not preflight.valid:
            store.flush_batch()
            return _finalize(
                store,
                manifest,
                accumulator,
                "validation_failed",
                total_examples,
                run_id,
                run_dir,
            )

        adapter = _build_adapter(config)
        engine: ChessEngineService | None = None
        if config.metrics.engine.enabled:
            engine = ChessEngineService(config.metrics.engine)
            try:
                engine.start()
                manifest["engine"] = engine.identity
            except Exception as exc:
                store.write_error(
                    _error_record(
                        "engine_start_failed",
                        "engine",
                        "startup",
                        f"Stockfish startup failed: {type(exc).__name__}",
                    )
                )
                accumulator.operational_errors += 1
                store.flush_batch()
                with suppress(Exception):
                    await adapter.aclose()
                engine.close()
                return _finalize(
                    store,
                    manifest,
                    accumulator,
                    "failed",
                    total_examples,
                    run_id,
                    run_dir,
                )

        manifest["status"] = "running"
        store.write_manifest(manifest)
        try:
            await _process_snapshot(
                config,
                run_id,
                store,
                task,
                suite,
                accumulator,
                adapter,
                engine,
                manifest,
                total_examples,
            )
        except asyncio.CancelledError:
            return _finalize(
                store,
                manifest,
                accumulator,
                "interrupted",
                total_examples,
                run_id,
                run_dir,
            )
        except Exception as exc:
            store.write_error(
                _error_record(
                    "run_failed",
                    "run",
                    "execution",
                    f"run stopped: {type(exc).__name__}",
                )
            )
            accumulator.operational_errors += 1
            store.flush_batch()
            return _finalize(
                store,
                manifest,
                accumulator,
                "failed",
                total_examples,
                run_id,
                run_dir,
            )
        finally:
            with suppress(Exception):
                await adapter.aclose()
            if engine is not None:
                engine.close()

        final_status: RunStatus = "partial" if accumulator.operational_errors else "complete"
        return _finalize(
            store,
            manifest,
            accumulator,
            final_status,
            total_examples,
            run_id,
            run_dir,
        )
    finally:
        store.close()


async def _process_snapshot(
    config: RunConfig,
    run_id: str,
    store: ArtifactStore,
    task: ChessFirstMoveTask,
    suite: ChessMetricSuite,
    accumulator: SummaryAccumulator,
    adapter: ModelAdapter,
    engine: ChessEngineService | None,
    manifest: dict[str, Any],
    total_examples: int,
) -> None:
    batch_size = config.model.batch_size
    batch: list[NormalizedExample] = []
    for example in JsonlSource.read_snapshot(store.snapshot_path):
        batch.append(example)
        if len(batch) == batch_size:
            await _process_batch(
                config, run_id, store, task, suite, accumulator, adapter, engine, batch
            )
            batch = []
            _checkpoint_manifest(store, manifest, accumulator, total_examples)
    if batch:
        await _process_batch(
            config, run_id, store, task, suite, accumulator, adapter, engine, batch
        )
        _checkpoint_manifest(store, manifest, accumulator, total_examples)


async def _process_batch(
    config: RunConfig,
    run_id: str,
    store: ArtifactStore,
    task: ChessFirstMoveTask,
    suite: ChessMetricSuite,
    accumulator: SummaryAccumulator,
    adapter: ModelAdapter,
    engine: ChessEngineService | None,
    examples: Sequence[NormalizedExample],
) -> None:
    requests: list[GenerationRequest] = []
    messages_by_id: dict[str, tuple[Any, ...]] = {}
    parameters = config.model.parameters.request_dict()
    for example in examples:
        messages = task.build_messages(example)
        request_id = sha256_text(f"{run_id}:{example.id}")[:32]
        requests.append(
            GenerationRequest(
                request_id=request_id,
                example_id=example.id,
                messages=messages,
                parameters=parameters,
            )
        )
        messages_by_id[request_id] = messages

    try:
        outcomes = await adapter.generate_batch(requests)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        outcomes = [
            GenerationError(
                request_id=request.request_id,
                example_id=request.example_id,
                code="adapter_failure",
                transient=False,
                attempts=1,
                message=f"adapter batch failed: {type(exc).__name__}",
                latency_ms=0,
            )
            for request in requests
        ]
    correlated = _correlate_outcomes(requests, outcomes)

    for example, request in zip(examples, requests, strict=True):
        generation = correlated[request.request_id]
        generation = _record_generation_error(store, generation)
        interpretation: MoveInterpretation | None = None
        if isinstance(generation, GenerationSuccess):
            interpretation = task.interpret(example, generation.text)
        bundle = suite.evaluate(example, generation, interpretation, engine)
        engine_error_id = None
        if bundle.engine_error:
            engine_record = _error_record(
                "engine_analysis_failed",
                "engine",
                "metric",
                bundle.engine_error,
                example_id=example.id,
                request_id=request.request_id,
            )
            engine_error_id = engine_record.error_id
            store.write_error(engine_record)

        result = _example_record(
            config,
            run_id,
            example,
            request,
            messages_by_id[request.request_id],
            generation,
            interpretation,
            bundle,
            engine_error_id,
        )
        store.write_example(result)
        accumulator.add(
            generation,
            interpretation,
            bundle.metrics,
            forced_move=bundle.forced_move,
        )
    store.flush_batch()


def _correlate_outcomes(
    requests: Sequence[GenerationRequest], outcomes: Sequence[GenerationOutcome]
) -> dict[str, GenerationOutcome]:
    expected = {request.request_id: request for request in requests}
    correlated: dict[str, GenerationOutcome] = {}
    invalid = False
    for outcome in outcomes:
        if (
            outcome.request_id not in expected
            or outcome.request_id in correlated
            or outcome.example_id != expected[outcome.request_id].example_id
        ):
            invalid = True
            break
        correlated[outcome.request_id] = outcome
    if invalid or set(correlated) != set(expected):
        return {
            request.request_id: GenerationError(
                request_id=request.request_id,
                example_id=request.example_id,
                code="adapter_contract_error",
                transient=False,
                attempts=1,
                message="adapter did not return exactly one correlated outcome per request",
                latency_ms=0,
            )
            for request in requests
        }
    return correlated


def _record_generation_error(store: ArtifactStore, outcome: GenerationOutcome) -> GenerationOutcome:
    if isinstance(outcome, GenerationSuccess):
        return outcome
    error = _error_record(
        outcome.code,
        "provider",
        "generation",
        outcome.message,
        example_id=outcome.example_id,
        request_id=outcome.request_id,
        transient=outcome.transient,
        attempts=outcome.attempts,
    )
    store.write_error(error)
    return outcome.model_copy(update={"error_id": error.error_id})


def _example_record(
    config: RunConfig,
    run_id: str,
    example: NormalizedExample,
    request: GenerationRequest,
    messages: tuple[Any, ...],
    generation: GenerationOutcome,
    interpretation: MoveInterpretation | None,
    bundle: EvaluationBundle,
    engine_error_id: str | None,
) -> dict[str, Any]:
    prompt_payload = [message.model_dump(mode="json") for message in messages]
    generation_payload = generation.model_dump(mode="json")
    raw_response = None
    response_hash = None
    if isinstance(generation, GenerationSuccess):
        raw_response = generation.text if config.artifacts.store_raw_responses else None
        response_hash = sha256_text(generation.text)
        generation_payload.pop("text", None)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "example_id": example.id,
        "input": example.input.model_dump(mode="json"),
        "reference": (
            example.reference.model_dump(mode="json") if example.reference is not None else None
        ),
        "metadata": example.metadata,
        "prompt": prompt_payload if config.artifacts.store_prompts else None,
        "prompt_hash": sha256_text(canonical_json(prompt_payload)),
        "prompt_version": config.task.prompt_version,
        "generation_parameters": request.parameters,
        "generation": generation_payload,
        "raw_response": raw_response,
        "response_hash": response_hash,
        "interpretation": interpretation.model_dump(mode="json") if interpretation else None,
        "engine_assessment": (
            bundle.engine_assessment.model_dump(mode="json")
            if bundle.engine_assessment is not None
            else None
        ),
        "engine_error_id": engine_error_id,
        "metrics": [metric.model_dump(mode="json") for metric in bundle.metrics],
    }


def _build_adapter(config: RunConfig) -> ModelAdapter:
    if isinstance(config.model, MockModelSettings):
        return MockModelAdapter(config.model)
    if isinstance(config.model, OpenAIChatSettings):
        return OpenAIChatAdapter(config.model, seed=config.run.seed)
    raise AssertionError("unsupported validated model kind")


def _initial_manifest(config: RunConfig, run_id: str, started_at: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package": {"name": "firstmove-eval", "version": __version__},
        "run_id": run_id,
        "status": "preflight",
        "started_at": started_at,
        "ended_at": None,
        "config": config.safe_manifest_dict(),
        "task": {
            "name": "chess_first_move",
            "version": "1",
            "prompt_version": config.task.prompt_version,
            "prompt_template_hash": ChessFirstMoveTask.prompt_template_hash,
        },
        "dataset": None,
        "engine": None,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "dependencies": _dependency_versions(),
            "git_revision": _git_revision(),
        },
        "counts": {"total": 0, "processed": 0, "operational_errors": 0},
    }


def _checkpoint_manifest(
    store: ArtifactStore,
    manifest: dict[str, Any],
    accumulator: SummaryAccumulator,
    total_examples: int,
) -> None:
    manifest["counts"] = {
        "total": total_examples,
        "processed": accumulator.processed,
        "operational_errors": accumulator.operational_errors,
    }
    store.write_manifest(manifest)


def _finalize(
    store: ArtifactStore,
    manifest: dict[str, Any],
    accumulator: SummaryAccumulator,
    status: RunStatus,
    total_examples: int,
    run_id: str,
    run_dir: Path,
) -> RunReport:
    manifest["status"] = status
    manifest["ended_at"] = _timestamp()
    manifest["counts"] = {
        "total": total_examples,
        "processed": accumulator.processed,
        "operational_errors": accumulator.operational_errors,
    }
    store.write_summary(accumulator.summary(total_examples, status))
    store.write_manifest(manifest)
    store.flush_batch()
    return RunReport(
        run_id=run_id,
        status=status,
        output_dir=run_dir,
        total_examples=total_examples,
        processed_examples=accumulator.processed,
        operational_errors=accumulator.operational_errors,
    )


def _error_record(
    code: str,
    category: Literal["validation", "provider", "engine", "run", "artifact"],
    stage: str,
    message: str,
    **details: Any,
) -> ErrorRecord:
    return ErrorRecord(
        error_id=f"err-{secrets.token_hex(8)}",
        code=code,
        category=category,
        stage=stage,
        message=message,
        **details,
    )


def _run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{secrets.token_hex(4)}"


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("chess", "httpx", "pydantic", "PyYAML", "typer"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision or None
