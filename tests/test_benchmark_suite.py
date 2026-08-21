from __future__ import annotations

from pathlib import Path

import chess
import pytest

import firstmove_eval.benchmarking as benchmarking_module
from benchmarks.suite import ScenarioSpec, compare_against_baseline, measure_scenario
from firstmove_eval.benchmark_plugins import benchmark_cases
from firstmove_eval.benchmarking import (
    BenchmarkContext,
    BenchmarkMeasurement,
    load_benchmark_config,
    load_benchmark_plugin,
    plugin_descriptors,
)
from firstmove_eval.task import ChessFirstMoveTask


def test_benchmark_corpus_is_legal_and_covers_both_sides() -> None:
    task = ChessFirstMoveTask()
    cases = benchmark_cases()
    assert {chess.Board(case.fen).turn for case in cases} == {chess.WHITE, chess.BLACK}
    for index, case in enumerate(cases):
        row = {
            "schema_version": 1,
            "id": f"case-{index}",
            "input": {"fen": case.fen, "variant": "standard"},
            "reference": {"move": case.reference},
            "metadata": {},
        }
        example = task.normalize_row(row)
        interpretation = task.interpret(example, case.response)
        assert interpretation.legal
        assert example.reference is not None
        assert interpretation.predicted_uci == example.reference.uci


@pytest.mark.integration
@pytest.mark.parametrize("scenario", ["hot_path", "preflight", "offline_run"])
def test_each_benchmark_scenario_smoke(scenario: str, tmp_path: Path) -> None:
    result = measure_scenario(
        scenario,
        ScenarioSpec(work_items=10, repetitions=1),
        tmp_path / scenario,
    )
    assert result["name"] == scenario
    assert result["throughput_per_second"] > 0


def test_baseline_comparison_reports_only_material_slowdowns() -> None:
    baseline = {
        "benchmarks": [
            {"name": "hot_path", "work_items": 100, "throughput_per_second": 1_000},
            {"name": "preflight", "work_items": 100, "throughput_per_second": 500},
        ]
    }
    current = {
        "benchmarks": [
            {"name": "hot_path", "work_items": 100, "throughput_per_second": 850},
            {"name": "preflight", "work_items": 100, "throughput_per_second": 350},
        ]
    }
    assert compare_against_baseline(current, baseline, 20) == [
        "preflight: 30.0% slower than baseline (allowed 20.0%)"
    ]


def test_baseline_comparison_rejects_different_plugin_options() -> None:
    baseline = {
        "benchmarks": [
            {
                "name": "offline",
                "plugin": "offline_run",
                "options_sha256": "baseline",
                "work_items": 100,
                "throughput_per_second": 500,
            }
        ]
    }
    current = {
        "benchmarks": [
            {
                "name": "offline",
                "plugin": "offline_run",
                "options_sha256": "candidate",
                "work_items": 100,
                "throughput_per_second": 500,
            }
        ]
    }
    assert compare_against_baseline(current, baseline, 20) == [
        "offline: plugin options differ from baseline"
    ]


def test_yaml_config_selects_plugins_and_resolves_result_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "benchmarks.yaml"
    config_path.write_text(
        """
schema_version: 1
name: pull-request
run:
  output: results/candidate.json
  baseline: results/main.json
  max_slowdown_percent: 15
benchmarks:
  - plugin: offline_run
    name: offline-small-batches
    work_items: 25
    repetitions: 2
    options:
      batch_size: 4
    max_peak_rss_mib: 96
""",
        encoding="utf-8",
    )

    config = load_benchmark_config(config_path)
    assert config.name == "pull-request"
    assert config.run.output == (tmp_path / "results/candidate.json").resolve()
    assert config.run.baseline == (tmp_path / "results/main.json").resolve()
    assert config.benchmarks[0].result_name == "offline-small-batches"
    assert config.benchmarks[0].options == {"batch_size": 4}


def test_builtin_plugins_are_discoverable_without_loading_external_code() -> None:
    descriptors = {descriptor.name: descriptor for descriptor in plugin_descriptors()}
    assert {"hot_path", "preflight", "offline_run"} <= descriptors.keys()
    assert descriptors["preflight"].source == "built-in"


def test_builtin_plugin_rejects_unknown_options(tmp_path: Path) -> None:
    plugin = load_benchmark_plugin("preflight")
    with pytest.raises(ValueError, match="unknown benchmark options: batch_sze"):
        plugin.measure(
            BenchmarkContext(
                root=tmp_path,
                work_items=1,
                repetitions=1,
                options={"batch_sze": 4},
            )
        )


def test_installed_entry_point_plugin_uses_public_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class ExamplePlugin:
        description = "Example third-party benchmark"

        def measure(self, context: BenchmarkContext) -> BenchmarkMeasurement:
            assert context.options == {"mode": "example"}
            return BenchmarkMeasurement((0.01,) * context.repetitions)

    class FakeEntryPoint:
        name = "example_plugin"
        value = "example_package:benchmark"
        dist = None

        @staticmethod
        def load() -> type[ExamplePlugin]:
            return ExamplePlugin

    monkeypatch.setattr(
        benchmarking_module,
        "_external_entry_points",
        lambda: (FakeEntryPoint(),),
    )

    plugin = load_benchmark_plugin("example_plugin")
    result = plugin.measure(
        BenchmarkContext(
            root=tmp_path,
            work_items=10,
            repetitions=2,
            options={"mode": "example"},
        )
    )
    assert result.samples_seconds == (0.01, 0.01)
