from __future__ import annotations

from typing import cast

from firstmove_eval.config import EngineSettings, MetricsSettings
from firstmove_eval.engine import ChessEngineService
from firstmove_eval.errors import EngineEvaluationError
from firstmove_eval.metrics import ChessMetricSuite, SummaryAccumulator
from firstmove_eval.models import (
    EngineAssessment,
    EngineScore,
    GenerationError,
    GenerationSuccess,
)
from firstmove_eval.task import ChessFirstMoveTask

from .conftest import source_row


def success(text: str = "e2e4") -> GenerationSuccess:
    return GenerationSuccess(
        request_id="request",
        example_id="example-1",
        text=text,
        latency_ms=1,
    )


class FakeEngine:
    def __init__(self, assessment: EngineAssessment | Exception) -> None:
        self.assessment = assessment

    def assess(self, fen: str, predicted_uci: str) -> EngineAssessment:
        del fen, predicted_uci
        if isinstance(self.assessment, Exception):
            raise self.assessment
        return self.assessment


def engine_metrics() -> MetricsSettings:
    return MetricsSettings(
        reference=True,
        engine=EngineSettings(enabled=True, path="stockfish"),
    )


def test_invalid_model_answer_is_measured_incorrect_not_skipped() -> None:
    task = ChessFirstMoveTask()
    example = task.normalize_row(source_row())
    generation = success("not-a-move")
    interpretation = task.interpret(example, generation.text)
    suite = ChessMetricSuite(engine_metrics())
    bundle = suite.evaluate(example, generation, interpretation, None)
    by_name = {metric.name: metric for metric in bundle.metrics}
    assert by_name["move_parse_success"].value is False
    assert by_name["reference_move_accuracy"].status == "measured"
    assert by_name["reference_move_accuracy"].value is False
    assert by_name["engine_best_move_match"].status == "measured"
    assert by_name["engine_best_move_match"].value is False
    assert by_name["centipawn_loss"].status == "skipped"


def test_generation_error_skips_quality_metrics() -> None:
    example = ChessFirstMoveTask().normalize_row(source_row())
    generation = GenerationError(
        request_id="request",
        example_id=example.id,
        code="provider_timeout",
        transient=True,
        attempts=3,
        message="timed out",
        latency_ms=100,
    )
    suite = ChessMetricSuite(engine_metrics())
    bundle = suite.evaluate(example, generation, None, None)
    assert all(metric.status == "skipped" for metric in bundle.metrics)
    assert {metric.reason for metric in bundle.metrics} == {"generation_error"}


def test_forced_move_count_is_independent_of_generation_success() -> None:
    task = ChessFirstMoveTask()
    example = task.normalize_row(
        source_row(
            fen="8/8/8/8/8/8/8/k1KQ4 b - - 0 1",
            reference="Ka2",
        )
    )
    generation = GenerationError(
        request_id="request",
        example_id=example.id,
        code="provider_timeout",
        transient=True,
        attempts=3,
        message="timed out",
        latency_ms=100,
    )
    suite = ChessMetricSuite(MetricsSettings())
    bundle = suite.evaluate(example, generation, None, None)
    accumulator = SummaryAccumulator(suite.metric_kinds)
    accumulator.add(generation, None, bundle.metrics, forced_move=bundle.forced_move)
    assert accumulator.summary(1, "partial")["forced_moves"] == 1


def test_missing_reference_is_skipped() -> None:
    task = ChessFirstMoveTask()
    example = task.normalize_row(source_row(reference=None))
    generation = success()
    bundle = ChessMetricSuite(MetricsSettings()).evaluate(
        example, generation, task.interpret(example, generation.text), None
    )
    reference = next(
        metric for metric in bundle.metrics if metric.name == "reference_move_accuracy"
    )
    assert reference.status == "skipped"
    assert reference.reason == "missing_reference"


def test_engine_cp_and_mate_semantics() -> None:
    task = ChessFirstMoveTask()
    example = task.normalize_row(source_row())
    generation = success()
    interpretation = task.interpret(example, generation.text)
    cp_assessment = EngineAssessment(
        forced_move=False,
        best_move_uci="d2d4",
        predicted_move_uci="e2e4",
        best_score=EngineScore(kind="cp", value=30),
        predicted_score=EngineScore(kind="cp", value=40),
        negative_loss_clamped=True,
    )
    suite = ChessMetricSuite(engine_metrics())
    cp_bundle = suite.evaluate(
        example,
        generation,
        interpretation,
        cast(ChessEngineService, FakeEngine(cp_assessment)),
    )
    cp = next(metric for metric in cp_bundle.metrics if metric.name == "centipawn_loss")
    assert cp.status == "measured"
    assert cp.value == 0
    assert cp.details["negative_loss_clamped"] is True

    mate_assessment = cp_assessment.model_copy(
        update={"best_score": EngineScore(kind="mate", value=3)}
    )
    mate_bundle = suite.evaluate(
        example,
        generation,
        interpretation,
        cast(ChessEngineService, FakeEngine(mate_assessment)),
    )
    mate = next(metric for metric in mate_bundle.metrics if metric.name == "centipawn_loss")
    assert mate.status == "skipped"
    assert mate.reason == "mate_score"


def test_engine_failure_marks_both_engine_metrics_error_once_in_summary() -> None:
    task = ChessFirstMoveTask()
    example = task.normalize_row(source_row())
    generation = success()
    interpretation = task.interpret(example, generation.text)
    suite = ChessMetricSuite(engine_metrics())
    bundle = suite.evaluate(
        example,
        generation,
        interpretation,
        cast(ChessEngineService, FakeEngine(EngineEvaluationError("failed"))),
    )
    engine_results = [metric for metric in bundle.metrics if metric.name.startswith("engine_")]
    cpl = next(metric for metric in bundle.metrics if metric.name == "centipawn_loss")
    assert engine_results[0].status == "error"
    assert cpl.status == "error"
    accumulator = SummaryAccumulator(suite.metric_kinds)
    accumulator.add(generation, interpretation, bundle.metrics, forced_move=bundle.forced_move)
    assert accumulator.operational_errors == 1


def test_agreement_matrix_uses_only_jointly_measured_booleans() -> None:
    task = ChessFirstMoveTask()
    example = task.normalize_row(source_row())
    generation = success()
    interpretation = task.interpret(example, generation.text)
    assessment = EngineAssessment(
        forced_move=False,
        best_move_uci="e2e4",
        predicted_move_uci="e2e4",
        best_score=EngineScore(kind="cp", value=20),
        predicted_score=EngineScore(kind="cp", value=20),
    )
    suite = ChessMetricSuite(engine_metrics())
    bundle = suite.evaluate(
        example,
        generation,
        interpretation,
        cast(ChessEngineService, FakeEngine(assessment)),
    )
    accumulator = SummaryAccumulator(suite.metric_kinds)
    accumulator.add(generation, interpretation, bundle.metrics, forced_move=bundle.forced_move)
    summary = accumulator.summary(1, "complete")
    assert summary["reference_engine_agreement"]["eligible"] == 1
    assert summary["reference_engine_agreement"]["reference_true_engine_true"] == 1
