"""Metric evaluation and constant-memory summary accumulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import chess

from firstmove_eval.config import MetricsSettings
from firstmove_eval.engine import ChessEngineService
from firstmove_eval.errors import EngineEvaluationError
from firstmove_eval.models import (
    EngineAssessment,
    GenerationError,
    GenerationOutcome,
    MetricResult,
    MoveInterpretation,
    NormalizedExample,
)


@dataclass(frozen=True, slots=True)
class EvaluationBundle:
    metrics: tuple[MetricResult, ...]
    engine_assessment: EngineAssessment | None = None
    engine_error: str | None = None
    forced_move: bool = False


class ChessMetricSuite:
    """Evaluate all V1 task metrics while sharing a single engine assessment."""

    def __init__(self, settings: MetricsSettings) -> None:
        self.settings = settings

    @property
    def metric_kinds(self) -> dict[str, Literal["boolean", "numeric"]]:
        result: dict[str, Literal["boolean", "numeric"]] = {
            "response_format_compliance": "boolean",
            "move_parse_success": "boolean",
            "legal_move_rate": "boolean",
        }
        if self.settings.reference:
            result["reference_move_accuracy"] = "boolean"
        if self.settings.engine.enabled:
            result["engine_best_move_match"] = "boolean"
            result["centipawn_loss"] = "numeric"
        return result

    @property
    def metric_definitions(self) -> dict[str, dict[str, str]]:
        return {
            name: {
                "value_type": kind,
                "aggregation": "mean",
                "unit": "centipawns" if kind == "numeric" else "proportion",
                "version": "1",
            }
            for name, kind in self.metric_kinds.items()
        }

    def evaluate(
        self,
        example: NormalizedExample,
        generation: GenerationOutcome,
        interpretation: MoveInterpretation | None,
        engine: ChessEngineService | None,
    ) -> EvaluationBundle:
        forced_move = chess.Board(example.input.fen).legal_moves.count() == 1
        if isinstance(generation, GenerationError):
            return EvaluationBundle(
                metrics=tuple(
                    MetricResult(name=name, status="skipped", reason="generation_error")
                    for name in self.metric_kinds
                ),
                forced_move=forced_move,
            )
        if interpretation is None:
            raise ValueError("successful generation requires an interpretation")

        metrics: list[MetricResult] = [
            _measured("response_format_compliance", interpretation.format_compliant),
            _measured("move_parse_success", interpretation.parse_success),
            _measured("legal_move_rate", interpretation.legal),
        ]
        if self.settings.reference:
            if example.reference is None:
                metrics.append(
                    MetricResult(
                        name="reference_move_accuracy",
                        status="skipped",
                        reason="missing_reference",
                    )
                )
            else:
                metrics.append(
                    _measured(
                        "reference_move_accuracy",
                        interpretation.legal
                        and interpretation.predicted_uci == example.reference.uci,
                    )
                )

        if not self.settings.engine.enabled:
            return EvaluationBundle(metrics=tuple(metrics), forced_move=forced_move)

        if not interpretation.legal or interpretation.predicted_uci is None:
            reason = "illegal_move" if interpretation.parse_success else "unparseable_move"
            metrics.extend(
                (
                    _measured("engine_best_move_match", False, {"reason": reason}),
                    MetricResult(name="centipawn_loss", status="skipped", reason=reason),
                )
            )
            return EvaluationBundle(metrics=tuple(metrics), forced_move=forced_move)

        if engine is None:
            raise ValueError("enabled engine metrics require an engine service")
        try:
            assessment = engine.assess(example.input.fen, interpretation.predicted_uci)
        except EngineEvaluationError as exc:
            metrics.extend(
                (
                    MetricResult(
                        name="engine_best_move_match",
                        status="error",
                        reason="engine_error",
                    ),
                    MetricResult(
                        name="centipawn_loss",
                        status="error",
                        reason="engine_error",
                    ),
                )
            )
            return EvaluationBundle(
                metrics=tuple(metrics),
                engine_error=str(exc),
                forced_move=forced_move,
            )

        metrics.append(
            _measured(
                "engine_best_move_match",
                assessment.best_move_uci == assessment.predicted_move_uci,
            )
        )
        score_details = {
            "best_score": assessment.best_score.model_dump(mode="json"),
            "predicted_score": assessment.predicted_score.model_dump(mode="json"),
        }
        if assessment.best_score.kind == "cp" and assessment.predicted_score.kind == "cp":
            raw_loss = assessment.best_score.value - assessment.predicted_score.value
            metrics.append(
                _measured(
                    "centipawn_loss",
                    max(0, raw_loss),
                    {
                        **score_details,
                        "negative_loss_clamped": raw_loss < 0,
                        "unit": "centipawns",
                    },
                )
            )
        else:
            metrics.append(
                MetricResult(
                    name="centipawn_loss",
                    status="skipped",
                    reason="mate_score",
                    details=score_details,
                )
            )
        return EvaluationBundle(
            metrics=tuple(metrics),
            engine_assessment=assessment,
            forced_move=forced_move,
        )


def _measured(
    name: str,
    value: bool | int | float,
    details: dict[str, Any] | None = None,
) -> MetricResult:
    return MetricResult(name=name, status="measured", value=value, details=details or {})


class SummaryAccumulator:
    """Constant-size aggregation with explicit coverage and agreement counts."""

    def __init__(self, metric_kinds: dict[str, Literal["boolean", "numeric"]]) -> None:
        self.metric_kinds = metric_kinds
        self.processed = 0
        self.generation_success = 0
        self.generation_error = 0
        self.forced_moves = 0
        self.operational_errors = 0
        self.metrics: dict[str, dict[str, Any]] = {}
        for name, kind in metric_kinds.items():
            self.metrics[name] = {
                "kind": kind,
                "measured": 0,
                "skipped": 0,
                "error": 0,
                "true": 0,
                "sum": 0.0,
                "min": None,
                "max": None,
                "reasons": {},
            }
        self.agreement = {
            "reference_true_engine_true": 0,
            "reference_true_engine_false": 0,
            "reference_false_engine_true": 0,
            "reference_false_engine_false": 0,
            "eligible": 0,
        }

    def add(
        self,
        generation: GenerationOutcome,
        interpretation: MoveInterpretation | None,
        metric_results: tuple[MetricResult, ...],
        *,
        forced_move: bool,
    ) -> None:
        self.processed += 1
        if isinstance(generation, GenerationError):
            self.generation_error += 1
            self.operational_errors += 1
        else:
            self.generation_success += 1
        if forced_move:
            self.forced_moves += 1

        measured_booleans: dict[str, bool] = {}
        example_has_metric_error = False
        for result in metric_results:
            aggregate = self.metrics[result.name]
            aggregate[result.status] += 1
            if result.reason:
                reasons = aggregate["reasons"]
                reasons[result.reason] = reasons.get(result.reason, 0) + 1
            if result.status == "error":
                example_has_metric_error = True
            if result.status != "measured":
                continue
            if aggregate["kind"] == "boolean":
                value = bool(result.value)
                aggregate["true"] += int(value)
                measured_booleans[result.name] = value
            else:
                numeric_value = float(result.value)  # type: ignore[arg-type]
                aggregate["sum"] += numeric_value
                aggregate["min"] = (
                    numeric_value
                    if aggregate["min"] is None
                    else min(aggregate["min"], numeric_value)
                )
                aggregate["max"] = (
                    numeric_value
                    if aggregate["max"] is None
                    else max(aggregate["max"], numeric_value)
                )

        if example_has_metric_error:
            self.operational_errors += 1

        reference = measured_booleans.get("reference_move_accuracy")
        engine = measured_booleans.get("engine_best_move_match")
        if reference is not None and engine is not None:
            self.agreement["eligible"] += 1
            key = f"reference_{str(reference).lower()}_engine_{str(engine).lower()}"
            self.agreement[key] += 1

    def summary(self, total_examples: int, status: str) -> dict[str, Any]:
        metric_summary: dict[str, Any] = {}
        for name, aggregate in self.metrics.items():
            measured = aggregate["measured"]
            item = {
                "kind": aggregate["kind"],
                "measured": measured,
                "skipped": aggregate["skipped"],
                "error": aggregate["error"],
                "coverage": measured / self.processed if self.processed else 0.0,
                "reasons": aggregate["reasons"],
            }
            if aggregate["kind"] == "boolean":
                item["numerator"] = aggregate["true"]
                item["rate"] = aggregate["true"] / measured if measured else None
            else:
                item["sum"] = aggregate["sum"] if measured else None
                item["mean"] = aggregate["sum"] / measured if measured else None
                item["min"] = aggregate["min"]
                item["max"] = aggregate["max"]
                item["unit"] = "centipawns"
            metric_summary[name] = item
        return {
            "schema_version": 1,
            "status": status,
            "total_examples": total_examples,
            "processed_examples": self.processed,
            "generation": {
                "success": self.generation_success,
                "error": self.generation_error,
                "coverage": self.generation_success / self.processed if self.processed else 0.0,
            },
            "forced_moves": self.forced_moves,
            "operational_errors": self.operational_errors,
            "metrics": metric_summary,
            "reference_engine_agreement": self.agreement,
        }
