from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

import firstmove_eval.runner as runner_module
from firstmove_eval.adapters.base import ModelAdapter
from firstmove_eval.cli import app
from firstmove_eval.config import ArtifactSettings
from firstmove_eval.models import (
    ChatMessage,
    GenerationError,
    GenerationOutcome,
    GenerationRequest,
    GenerationSuccess,
)
from firstmove_eval.runner import _correlate_outcomes, _run_async, run, validate

from .conftest import source_row, write_rows


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_validate_and_complete_mock_run(mock_config: object) -> None:
    config = mock_config
    validation = validate(config)  # type: ignore[arg-type]
    assert validation.valid
    assert validation.example_count == 1

    report = run(config)  # type: ignore[arg-type]
    assert report.status == "complete"
    assert report.exit_code == 0
    assert {path.name for path in report.output_dir.iterdir()} == {
        "manifest.json",
        "input_snapshot.jsonl",
        "examples.jsonl",
        "summary.json",
        "errors.jsonl",
    }
    summary = read_json(report.output_dir / "summary.json")
    metrics = summary["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["reference_move_accuracy"]["rate"] == 1.0  # type: ignore[index]
    manifest_text = (report.output_dir / "manifest.json").read_text(encoding="utf-8")
    assert "status" in manifest_text
    assert "prompt_template_hash" in manifest_text


def test_invalid_dataset_never_builds_model_adapter(
    mock_config: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = mock_config
    write_rows(config.dataset.path, [source_row(), source_row()])  # type: ignore[attr-defined]

    def forbidden(_: object) -> ModelAdapter:
        raise AssertionError("adapter must not be built before preflight succeeds")

    monkeypatch.setattr(runner_module, "_build_adapter", forbidden)
    report = run(config)  # type: ignore[arg-type]
    assert report.status == "validation_failed"
    assert report.processed_examples == 0
    errors = read_jsonl(report.output_dir / "errors.jsonl")
    assert errors[0]["code"] == "duplicate_id"
    assert not (report.output_dir / "input_snapshot.jsonl").exists()


class ErrorAdapter:
    async def generate_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> list[GenerationOutcome]:
        return [
            GenerationError(
                request_id=request.request_id,
                example_id=request.example_id,
                code="provider_timeout",
                transient=True,
                attempts=3,
                message="provider request timed out",
                latency_ms=20,
            )
            for request in requests
        ]

    async def aclose(self) -> None:
        return None


def test_provider_error_finalizes_partial_and_skips_quality_denominators(
    mock_config: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_module, "_build_adapter", lambda _: ErrorAdapter())
    report = run(mock_config)  # type: ignore[arg-type]
    assert report.status == "partial"
    assert report.exit_code == 2
    assert report.operational_errors == 1
    summary = read_json(report.output_dir / "summary.json")
    assert summary["generation"] == {"coverage": 0.0, "error": 1, "success": 0}
    metrics = summary["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["reference_move_accuracy"]["measured"] == 0  # type: ignore[index]
    example = read_jsonl(report.output_dir / "examples.jsonl")[0]
    generation = example["generation"]
    assert isinstance(generation, dict)
    assert generation["error_id"]


def test_prompt_and_response_can_be_hash_only(mock_config: object) -> None:
    config = mock_config.model_copy(  # type: ignore[attr-defined]
        update={"artifacts": ArtifactSettings(store_prompts=False, store_raw_responses=False)}
    )
    report = run(config)
    example = read_jsonl(report.output_dir / "examples.jsonl")[0]
    assert example["prompt"] is None
    assert example["raw_response"] is None
    assert example["prompt_hash"]
    assert example["response_hash"]


def test_adapter_contract_violation_becomes_correlated_errors(
    mock_config: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    class EmptyAdapter:
        async def generate_batch(
            self, requests: Sequence[GenerationRequest]
        ) -> list[GenerationOutcome]:
            del requests
            return []

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(runner_module, "_build_adapter", lambda _: EmptyAdapter())
    report = run(mock_config)  # type: ignore[arg-type]
    assert report.status == "partial"
    errors = read_jsonl(report.output_dir / "errors.jsonl")
    assert errors[0]["code"] == "adapter_contract_error"


def test_reordered_outcomes_are_correlated_by_request_id() -> None:
    requests = [
        GenerationRequest(
            request_id=f"request-{index}",
            example_id=f"example-{index}",
            messages=(ChatMessage(role="user", content="position"),),
            parameters={},
        )
        for index in range(2)
    ]
    outcomes = [
        GenerationSuccess(
            request_id=request.request_id,
            example_id=request.example_id,
            text="e2e4",
            latency_ms=0,
        )
        for request in reversed(requests)
    ]
    correlated = _correlate_outcomes(requests, outcomes)
    assert list(correlated) == ["request-1", "request-0"]
    assert correlated["request-0"].example_id == "example-0"


@pytest.mark.asyncio
async def test_cancellation_finalizes_interrupted_artifacts(
    mock_config: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()

    class BlockingAdapter:
        async def generate_batch(
            self, requests: Sequence[GenerationRequest]
        ) -> list[GenerationOutcome]:
            del requests
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(runner_module, "_build_adapter", lambda _: BlockingAdapter())
    execution = asyncio.create_task(_run_async(mock_config))  # type: ignore[arg-type]
    await started.wait()
    execution.cancel()
    report = await execution
    assert report.status == "interrupted"
    assert report.exit_code == 130
    assert read_json(report.output_dir / "manifest.json")["status"] == "interrupted"
    assert read_json(report.output_dir / "summary.json")["processed_examples"] == 0


def test_cli_validate_and_run(tmp_path: Path) -> None:
    dataset = tmp_path / "positions.jsonl"
    write_rows(dataset, [source_row()])
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
schema_version: 1
run:
  output_dir: {str(tmp_path / "runs").replace("\\", "/")}
dataset:
  path: {str(dataset).replace("\\", "/")}
model:
  kind: mock
  default_response: e2e4
metrics:
  engine:
    enabled: false
""",
        encoding="utf-8",
    )
    cli = CliRunner()
    validation = cli.invoke(app, ["validate", "--config", str(config_path)])
    assert validation.exit_code == 0, validation.output
    execution = cli.invoke(app, ["run", "--config", str(config_path)])
    assert execution.exit_code == 0, execution.output
    assert '"status": "complete"' in execution.output
