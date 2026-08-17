# Contributing

Use Python 3.13 or later and install the development environment with `uv sync --extra dev`.
Before submitting a change, run:

```console
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
uv build
uv run twine check dist/*
```

Do not add a bundled Stockfish executable, live paid-provider call, dynamic component loader,
or heavyweight model dependency without a separately reviewed architecture change.

