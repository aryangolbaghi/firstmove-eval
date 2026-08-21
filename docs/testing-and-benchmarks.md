# Testing and benchmarks

The test suite has three lanes so the amount of feedback can match the change being made.

## Correctness tests

For the fastest edit-test loop, run the focused unit and property tests:

```console
uv run pytest -m "not integration and not stockfish"
```

Before opening a pull request, include the deterministic filesystem, HTTP, CLI, and full
evaluation-pipeline tests:

```console
uv run pytest -m "not stockfish"
```

`uv run pytest` remains the complete command. The real Stockfish test is selected only when
`STOCKFISH_PATH` points to a local executable. To find tests slowing down the feedback loop,
append `--durations=10` to any command.

The end-to-end regression corpus deliberately includes exact UCI, legal SAN, a legal but
incorrect move, and an unparseable response. It checks the resulting format, parsing,
legality, and reference-accuracy denominators together rather than testing those metrics only
in isolation.

## Performance benchmarks

The benchmark suite is offline and deterministic: it never calls a paid model or Stockfish.
Its three scenarios measure different sources of cost:

- `hot_path`: chess-row normalization plus model-response interpretation across positions for
  both sides to move.
- `preflight`: JSONL decoding, schema/chess validation, duplicate-ID tracking, hashing, and
  canonical snapshot writing.
- `offline_run`: the complete mock-model pipeline, including batching, metrics, checkpoints,
  and artifact writes.

The development extra already includes the benchmark runtime. Wheel users can install it with
`pip install "firstmove-eval[benchmark]"`.

List the built-in and installed third-party benchmark plugins:

```console
uv run firstmove-benchmark --list-plugins
```

Run the included YAML suite during development:

```console
uv run firstmove-benchmark --config examples/benchmarks.quick.yaml
```

The configuration is intentionally small:

```yaml
schema_version: 1
name: quick-config
run:
  output: ../benchmark-results/quick-config.json
  max_slowdown_percent: 20
benchmarks:
  - plugin: preflight
    work_items: 1000
    repetitions: 3
    max_peak_rss_mib: 128
  - plugin: offline_run
    name: offline-small-batches
    work_items: 250
    repetitions: 2
    options:
      batch_size: 32
```

`plugin` selects a built-in or installed plugin. The optional `name` creates a distinct result
name, allowing the same plugin to appear more than once with different settings. `options` is
a JSON-safe mapping interpreted by that plugin; the built-in `offline_run` plugin currently
accepts `batch_size`. Output and baseline paths are resolved relative to the YAML file.

The original profile commands remain convenient compatibility presets:

```console
uv run firstmove-benchmark --profile quick
uv run firstmove-benchmark --profile scale
uv run firstmove-benchmark --scenario preflight --work-items 10000 --repetitions 3
```

For a regression gate, first record a baseline from the target branch on the same machine.
Then set `run.baseline` in the YAML or override it on the command line:

```console
uv run firstmove-benchmark --config examples/benchmarks.quick.yaml --output benchmark-results/main.json
uv run firstmove-benchmark --config examples/benchmarks.quick.yaml --baseline benchmark-results/main.json --output benchmark-results/candidate.json
```

The command exits nonzero if a matching scenario loses more throughput than the allowed
percentage. Comparing runs from the same machine and power mode is important; absolute timing
thresholds are intentionally avoided because they are unreliable across developer and CI
hardware. Results record a hash rather than the raw plugin options, and comparison refuses to
compare jobs whose plugin, options hash, or work-item count differs.

Use the scale profile for release evidence:

```console
uv run firstmove-benchmark --profile scale --output benchmark-results/scale.json
```

The scale preset keeps the existing 100,000-row preflight target and fails if its worker
exceeds 128 MiB peak resident memory. Dataset generation happens before timed preflight and
offline-run samples. Each scenario runs in a fresh spawned process, and the parent records its
peak resident memory. Timing output reports every raw sample, the median, and median-derived
throughput in the JSON artifact.

## Third-party benchmark plugins

A plugin package registers a standard Python entry point. It does not require changes to
FirstMove Eval's runner:

```toml
[project.entry-points."firstmove_eval.benchmarks"]
my_benchmark = "my_package.benchmark:benchmark"
```

The referenced object implements the public contract:

```python
import time

from firstmove_eval.benchmarking import (
    BenchmarkContext,
    BenchmarkMeasurement,
)


class MyBenchmark:
    description = "Measure my application operation"

    def measure(self, context: BenchmarkContext) -> BenchmarkMeasurement:
        samples = []
        for _ in range(context.repetitions):
            started = time.perf_counter()
            for _ in range(context.work_items):
                run_one_operation(context.options)
            samples.append(time.perf_counter() - started)
        return BenchmarkMeasurement(tuple(samples))


benchmark = MyBenchmark()
```

After installing that package into the environment, select it like any built-in:

```yaml
benchmarks:
  - plugin: my_benchmark
    work_items: 500
    repetitions: 3
    options:
      mode: realistic
```

Discovery reads package metadata without importing third-party code. The selected plugin is
loaded only inside its isolated benchmark worker. As with any installed Python extension, a
selected plugin executes with the user's permissions, so install benchmark plugins only from
trusted packages.
