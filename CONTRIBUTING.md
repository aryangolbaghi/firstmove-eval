# Contributing

Use Python 3.13 or later and install the development environment with `uv sync --extra dev`.
Use the fast correctness lane while editing:

```console
uv run pytest -m "not integration and not stockfish"
```

Before submitting a change, run the deterministic integration lane and project checks:

```console
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -m "not stockfish"
uv build
uv run twine check dist/*
```

Performance-sensitive changes should also run
`uv run firstmove-benchmark --config examples/benchmarks.quick.yaml` and compare it with a
same-machine baseline. See [testing and benchmarks](docs/testing-and-benchmarks.md) for the
plugin configuration, extension contract, regression gate, and release-scale profile.

Do not add a bundled Stockfish executable, live paid-provider call, unrestricted module-path
loader, or heavyweight model dependency without a separately reviewed architecture change.
Benchmark extensions must use the named, installed entry-point contract.
