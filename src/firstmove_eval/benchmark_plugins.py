"""Built-in benchmark plugin implementations."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess

from firstmove_eval.benchmarking import (
    BenchmarkContext,
    BenchmarkMeasurement,
    BenchmarkPluginFactory,
)
from firstmove_eval.config import (
    DatasetSettings,
    EngineSettings,
    MetricsSettings,
    MockModelSettings,
    RunConfig,
    RunSettings,
)
from firstmove_eval.runner import run
from firstmove_eval.source import JsonlSource
from firstmove_eval.task import ChessFirstMoveTask


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    fen: str
    reference: str
    response: str


def benchmark_cases() -> tuple[BenchmarkCase, ...]:
    """Return deterministic positions spanning opening and middlegame, for both sides."""

    definitions = (
        ((), "e4"),
        (("e4",), "c5"),
        (("d4", "d5", "c4"), "e6"),
        (("e4", "e5", "Nf3", "Nc6", "Bb5", "a6"), "Ba4"),
        (("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3"), "a6"),
    )
    cases: list[BenchmarkCase] = []
    for played_moves, reference in definitions:
        board = chess.Board()
        for san in played_moves:
            board.push_san(san)
        move = board.parse_san(reference)
        cases.append(
            BenchmarkCase(
                fen=board.fen(en_passant="fen"),
                reference=reference,
                response=move.uci(),
            )
        )
    return tuple(cases)


def source_row(case: BenchmarkCase, example_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": example_id,
        "input": {"fen": case.fen, "variant": "standard"},
        "reference": {"move": case.reference, "source_line": [case.reference]},
        "metadata": {"source": "benchmark"},
    }


class HotPathBenchmark:
    description = "Chess row normalization and model-response interpretation"

    def measure(self, context: BenchmarkContext) -> BenchmarkMeasurement:
        _reject_unknown_options(context, set())
        cases = benchmark_cases()
        rows = tuple(source_row(case, f"hot-{index}") for index, case in enumerate(cases))
        task = ChessFirstMoveTask()

        for index in range(min(100, context.work_items)):
            case_index = index % len(cases)
            example = task.normalize_row(rows[case_index])
            task.interpret(example, cases[case_index].response)

        samples: list[float] = []
        for _ in range(context.repetitions):
            valid = 0
            started = time.perf_counter()
            for index in range(context.work_items):
                case_index = index % len(cases)
                example = task.normalize_row(rows[case_index])
                interpretation = task.interpret(example, cases[case_index].response)
                valid += int(interpretation.legal and interpretation.parse_success)
            samples.append(time.perf_counter() - started)
            if valid != context.work_items:
                raise RuntimeError("hot-path benchmark produced an invalid interpretation")
        return BenchmarkMeasurement(tuple(samples))


class PreflightBenchmark:
    description = "JSONL validation, duplicate tracking, hashing, and canonical snapshot writing"

    def measure(self, context: BenchmarkContext) -> BenchmarkMeasurement:
        _reject_unknown_options(context, set())
        dataset = context.root / "positions.jsonl"
        _write_dataset(dataset, context.work_items)
        task = ChessFirstMoveTask()
        source = JsonlSource(dataset)
        samples: list[float] = []
        for repetition in range(context.repetitions):
            snapshot = context.root / f"input_snapshot_{repetition}.jsonl"
            started = time.perf_counter()
            result = source.preflight(task, snapshot)
            samples.append(time.perf_counter() - started)
            if not result.valid or result.example_count != context.work_items:
                raise RuntimeError("preflight benchmark did not validate every input row")
        return BenchmarkMeasurement(tuple(samples))


class OfflineRunBenchmark:
    description = "Complete deterministic mock-model evaluation and artifact pipeline"

    def measure(self, context: BenchmarkContext) -> BenchmarkMeasurement:
        _reject_unknown_options(context, {"batch_size"})
        cases = benchmark_cases()
        dataset = context.root / "positions.jsonl"
        _write_dataset(dataset, context.work_items)
        batch_size = _positive_int_option(context, "batch_size", default=128)
        responses = {
            f"position-{index}": cases[index % len(cases)].response
            for index in range(context.work_items)
        }
        config = RunConfig(
            run=RunSettings(output_dir=context.root / "runs"),
            dataset=DatasetSettings(path=dataset),
            model=MockModelSettings(batch_size=batch_size, responses=responses),
            metrics=MetricsSettings(engine=EngineSettings(enabled=False)),
        )

        samples: list[float] = []
        for _ in range(context.repetitions):
            started = time.perf_counter()
            report = run(config)
            samples.append(time.perf_counter() - started)
            if report.status != "complete" or report.processed_examples != context.work_items:
                raise RuntimeError("offline-run benchmark did not complete every input row")
            summary = json.loads((report.output_dir / "summary.json").read_text(encoding="utf-8"))
            if summary["metrics"]["reference_move_accuracy"]["rate"] != 1.0:
                raise RuntimeError("offline-run benchmark produced incorrect reference metrics")
        return BenchmarkMeasurement(tuple(samples))


def _write_dataset(path: Path, work_items: int) -> None:
    cases = benchmark_cases()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(work_items):
            row = source_row(cases[index % len(cases)], f"position-{index}")
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _positive_int_option(context: BenchmarkContext, name: str, default: int) -> int:
    value = context.options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} benchmark option must be a positive integer")
    return value


def _reject_unknown_options(context: BenchmarkContext, allowed: set[str]) -> None:
    unknown = sorted(context.options.keys() - allowed)
    if unknown:
        raise ValueError(f"unknown benchmark options: {', '.join(unknown)}")


BUILTIN_BENCHMARK_PLUGINS: dict[str, BenchmarkPluginFactory] = {
    "hot_path": HotPathBenchmark,
    "preflight": PreflightBenchmark,
    "offline_run": OfflineRunBenchmark,
}
