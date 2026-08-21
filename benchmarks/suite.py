"""Configurable benchmark plugin runner with optional baseline regression gates."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import statistics
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty
from typing import Any

from firstmove_eval._json import canonical_json, sha256_text
from firstmove_eval.benchmarking import (
    BenchmarkContext,
    BenchmarkJobConfig,
    BenchmarkPluginError,
    BenchmarkRunSettings,
    BenchmarkSuiteConfig,
    load_benchmark_config,
    load_benchmark_plugin,
    plugin_descriptors,
)
from firstmove_eval.errors import ConfigurationError

SCENARIO_NAMES = ("hot_path", "preflight", "offline_run")
MAX_SCALE_PREFLIGHT_RSS_MIB = 128.0


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    work_items: int
    repetitions: int


PROFILE_SPECS: dict[str, dict[str, ScenarioSpec]] = {
    "quick": {
        "hot_path": ScenarioSpec(work_items=5_000, repetitions=3),
        "preflight": ScenarioSpec(work_items=1_000, repetitions=3),
        "offline_run": ScenarioSpec(work_items=250, repetitions=2),
    },
    "scale": {
        "hot_path": ScenarioSpec(work_items=50_000, repetitions=5),
        "preflight": ScenarioSpec(work_items=100_000, repetitions=1),
        "offline_run": ScenarioSpec(work_items=10_000, repetitions=1),
    },
}


def measure_scenario(name: str, spec: ScenarioSpec, root: Path) -> dict[str, Any]:
    """Measure one plugin in-process; process isolation is added by the CLI."""

    job = BenchmarkJobConfig(
        plugin=name,
        work_items=spec.work_items,
        repetitions=spec.repetitions,
    )
    return _measure_job(job, root)


def compare_against_baseline(
    current: dict[str, Any], baseline: dict[str, Any], max_slowdown_percent: float
) -> list[str]:
    """Return human-readable failures for matching throughput measurements."""

    current_results = _results_by_name(current)
    baseline_results = _results_by_name(baseline)
    matching = current_results.keys() & baseline_results.keys()
    if not matching:
        return ["baseline contains no benchmark scenarios matching this run"]

    failures: list[str] = []
    for name in sorted(matching):
        current_plugin = current_results[name].get("plugin")
        baseline_plugin = baseline_results[name].get("plugin")
        if (
            current_plugin is not None
            and baseline_plugin is not None
            and current_plugin != baseline_plugin
        ):
            failures.append(
                f"{name}: plugin differs from baseline ({current_plugin} != {baseline_plugin})"
            )
            continue
        current_options = current_results[name].get("options_sha256")
        baseline_options = baseline_results[name].get("options_sha256")
        if (
            current_options is not None
            and baseline_options is not None
            and current_options != baseline_options
        ):
            failures.append(f"{name}: plugin options differ from baseline")
            continue
        current_items = current_results[name].get("work_items")
        baseline_items = baseline_results[name].get("work_items")
        if (
            current_items is not None
            and baseline_items is not None
            and current_items != baseline_items
        ):
            failures.append(
                f"{name}: work item count differs from baseline "
                f"({current_items} != {baseline_items})"
            )
            continue
        current_rate = float(current_results[name]["throughput_per_second"])
        baseline_rate = float(baseline_results[name]["throughput_per_second"])
        if baseline_rate <= 0:
            failures.append(f"{name}: baseline throughput must be greater than zero")
            continue
        slowdown = (baseline_rate - current_rate) / baseline_rate * 100
        if slowdown > max_slowdown_percent:
            failures.append(
                f"{name}: {slowdown:.1f}% slower than baseline "
                f"(allowed {max_slowdown_percent:.1f}%)"
            )
    return failures


def _measure_job(job: BenchmarkJobConfig, root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    plugin = load_benchmark_plugin(job.plugin)
    measurement = plugin.measure(
        BenchmarkContext(
            root=root,
            work_items=job.work_items,
            repetitions=job.repetitions,
            options=job.options,
        )
    )
    samples = measurement.samples_seconds
    if len(samples) != job.repetitions:
        raise BenchmarkPluginError(
            f"plugin {job.plugin!r} returned {len(samples)} samples; expected {job.repetitions}"
        )
    median_seconds = statistics.median(samples)
    return {
        "name": job.result_name,
        "plugin": job.plugin,
        "options_sha256": sha256_text(canonical_json(job.options)),
        "work_items": job.work_items,
        "repetitions": job.repetitions,
        "samples_seconds": [round(sample, 6) for sample in samples],
        "median_seconds": round(median_seconds, 6),
        "throughput_per_second": round(job.work_items / median_seconds, 2),
    }


def _scenario_worker(
    job_payload: dict[str, Any],
    root: str,
    results: multiprocessing.Queue[dict[str, Any]],
) -> None:
    try:
        job = BenchmarkJobConfig.model_validate(job_payload)
        result = _measure_job(job, Path(root))
    except Exception as exc:
        results.put({"error": f"{type(exc).__name__}: {exc}"})
    else:
        results.put(result)


def _run_monitored(job: BenchmarkJobConfig, root: Path) -> dict[str, Any]:
    psutil = _load_psutil()
    context = multiprocessing.get_context("spawn")
    results: multiprocessing.Queue[dict[str, Any]] = context.Queue()
    process = context.Process(
        target=_scenario_worker,
        args=(job.model_dump(mode="json"), str(root), results),
    )
    process.start()
    monitored = psutil.Process(process.pid)
    peak_bytes = 0
    while process.is_alive():
        with suppress(psutil.Error):
            peak_bytes = max(peak_bytes, monitored.memory_info().rss)
        time.sleep(0.01)
    process.join()
    try:
        result = results.get(timeout=2)
    except Empty as exc:
        raise BenchmarkPluginError(
            f"{job.result_name} worker failed with exit code {process.exitcode}"
        ) from exc
    finally:
        results.close()
    if "error" in result:
        raise BenchmarkPluginError(f"{job.result_name} worker failed: {result['error']}")
    result["peak_rss_mib"] = round(peak_bytes / (1024 * 1024), 2)
    return result


def _load_psutil() -> Any:
    try:
        import psutil
    except ImportError as exc:
        raise BenchmarkPluginError(
            "benchmark memory monitoring requires the 'benchmark' extra; "
            "install firstmove-eval[benchmark]"
        ) from exc
    return psutil


def _results_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_results = payload.get("benchmarks")
    if not isinstance(raw_results, list):
        raise ValueError("benchmark payload must contain a benchmarks list")
    return {
        str(result["name"]): result
        for result in raw_results
        if isinstance(result, dict) and "name" in result
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, help="YAML benchmark-suite configuration.")
    source.add_argument("--profile", choices=PROFILE_SPECS, help="Built-in compatibility preset.")
    parser.add_argument("--list-plugins", action="store_true", help="List available plugins.")
    parser.add_argument(
        "--scenario",
        action="append",
        help="Preset plugin to run; may be supplied more than once.",
    )
    parser.add_argument(
        "--work-items",
        type=int,
        help="Override work items when exactly one preset plugin is selected.",
    )
    parser.add_argument("--repetitions", type=int, help="Override preset repetitions.")
    parser.add_argument("--output", type=Path, help="Override the JSON result path.")
    parser.add_argument("--baseline", type=Path, help="Override the baseline result path.")
    parser.add_argument(
        "--max-slowdown-percent",
        type=float,
        help="Override the allowed throughput regression percentage.",
    )
    parser.add_argument(
        "--max-preflight-rss-mib",
        type=float,
        help="Override the preset preflight peak-memory gate.",
    )
    args = parser.parse_args(argv)
    if args.config and any(
        value is not None
        for value in (
            args.scenario,
            args.work_items,
            args.repetitions,
            args.max_preflight_rss_mib,
        )
    ):
        parser.error("--config cannot be combined with preset workload or preflight-memory options")
    if args.work_items is not None and (not args.scenario or len(args.scenario) != 1):
        parser.error("--work-items requires exactly one --scenario")
    for name in ("work_items", "repetitions"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.max_slowdown_percent is not None and args.max_slowdown_percent < 0:
        parser.error("--max-slowdown-percent cannot be negative")
    if args.max_preflight_rss_mib is not None and args.max_preflight_rss_mib <= 0:
        parser.error("--max-preflight-rss-mib must be greater than zero")
    return args


def _preset_config(args: argparse.Namespace) -> BenchmarkSuiteConfig:
    profile = args.profile or "quick"
    selected = tuple(dict.fromkeys(args.scenario or SCENARIO_NAMES))
    jobs: list[BenchmarkJobConfig] = []
    for plugin_name in selected:
        base_spec = PROFILE_SPECS[profile].get(plugin_name)
        if base_spec is None and args.work_items is None:
            raise ConfigurationError(f"custom preset plugin {plugin_name!r} requires --work-items")
        work_items = args.work_items or (base_spec.work_items if base_spec else 0)
        repetitions = args.repetitions or (base_spec.repetitions if base_spec else 1)
        memory_limit = None
        if plugin_name == "preflight":
            memory_limit = args.max_preflight_rss_mib
            if memory_limit is None and profile == "scale":
                memory_limit = MAX_SCALE_PREFLIGHT_RSS_MIB
        jobs.append(
            BenchmarkJobConfig(
                plugin=plugin_name,
                work_items=work_items,
                repetitions=repetitions,
                max_peak_rss_mib=memory_limit,
            )
        )
    return BenchmarkSuiteConfig(
        name=profile,
        run=BenchmarkRunSettings(
            output=args.output.resolve() if args.output else None,
            baseline=args.baseline.resolve() if args.baseline else None,
            max_slowdown_percent=(
                args.max_slowdown_percent if args.max_slowdown_percent is not None else 20.0
            ),
        ),
        benchmarks=tuple(jobs),
    )


def _configured_suite(args: argparse.Namespace) -> BenchmarkSuiteConfig:
    if args.config:
        config = load_benchmark_config(args.config)
        run_settings = config.run.model_copy(
            update={
                "output": args.output.resolve() if args.output else config.run.output,
                "baseline": args.baseline.resolve() if args.baseline else config.run.baseline,
                "max_slowdown_percent": (
                    args.max_slowdown_percent
                    if args.max_slowdown_percent is not None
                    else config.run.max_slowdown_percent
                ),
            }
        )
        return config.model_copy(update={"run": run_settings})
    return _preset_config(args)


def _execute_suite(config: BenchmarkSuiteConfig) -> tuple[dict[str, Any], list[str]]:
    available_plugins = {descriptor.name for descriptor in plugin_descriptors()}
    missing = sorted({job.plugin for job in config.benchmarks} - available_plugins)
    if missing:
        raise BenchmarkPluginError(
            f"benchmark plugins are not installed: {', '.join(missing)}; "
            f"use --list-plugins to inspect availability"
        )

    benchmark_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="firstmove-benchmarks-") as temporary:
        root = Path(temporary)
        for job in config.benchmarks:
            result = _run_monitored(job, root / job.result_name)
            benchmark_results.append(result)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "profile": config.name,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "benchmarks": benchmark_results,
    }

    failures: list[str] = []
    if config.run.baseline:
        baseline = json.loads(config.run.baseline.read_text(encoding="utf-8"))
        failures.extend(
            compare_against_baseline(payload, baseline, config.run.max_slowdown_percent)
        )
    by_name = _results_by_name(payload)
    for job in config.benchmarks:
        result = by_name[job.result_name]
        if (
            job.max_peak_rss_mib is not None
            and float(result["peak_rss_mib"]) > job.max_peak_rss_mib
        ):
            failures.append(
                f"{job.result_name}: peak RSS {result['peak_rss_mib']:.2f} MiB exceeded "
                f"{job.max_peak_rss_mib:.2f} MiB"
            )
    payload["failures"] = failures
    return payload, failures


def _print_plugins() -> None:
    print(f"{'plugin':<20} {'source':<24} description")
    for descriptor in plugin_descriptors():
        print(f"{descriptor.name:<20} {descriptor.source:<24} {descriptor.description}")


def _print_results(config: BenchmarkSuiteConfig, payload: dict[str, Any]) -> None:
    print(f"suite={config.name} python={payload['runtime']['python']}")
    print(f"{'benchmark':<18} {'items':>10} {'median_s':>10} {'items/s':>12} {'peak_mib':>10}")
    for result in payload["benchmarks"]:
        print(
            f"{result['name']:<18} {result['work_items']:>10,d} "
            f"{result['median_seconds']:>10.3f} "
            f"{result['throughput_per_second']:>12,.1f} "
            f"{result['peak_rss_mib']:>10.2f}"
        )
    for failure in payload["failures"]:
        print(f"FAIL: {failure}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list_plugins:
        _print_plugins()
        return 0
    try:
        config = _configured_suite(args)
        payload, failures = _execute_suite(config)
        _print_results(config, payload)
        if config.run.output:
            config.run.output.parent.mkdir(parents=True, exist_ok=True)
            config.run.output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"results={config.run.output}")
    except (BenchmarkPluginError, ConfigurationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
