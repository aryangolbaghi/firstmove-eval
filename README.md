# FirstMove Eval

FirstMove Eval is a reproducible command-line and Python evaluation tool for one-move
answers to standard-chess positions. It keeps benchmark-label agreement separate from
Stockfish agreement and preserves model and evaluator failures without corrupting metric
denominators.

## Requirements

- Python 3.13 or later
- A user-supplied Stockfish executable when engine metrics are enabled

Stockfish is not included in this package.

## Install

```console
pip install firstmove-eval
```

For development:

```console
uv sync --extra dev
```

## Input

Each nonblank line is a versioned JSON object:

```json
{"schema_version":1,"id":"start","input":{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","variant":"standard"},"reference":{"move":"e4","source_line":["e4"]},"metadata":{"split":"test"}}
```

References are optional. A supplied reference may be legal UCI or SAN and is normalized to
UCI during preflight. Invalid rows fail the entire preflight before any model request.

## Run

Copy `examples/config.mock.yaml`, adjust its dataset path, then run:

```console
firstmove-eval validate --config examples/config.mock.yaml
firstmove-eval run --config examples/config.mock.yaml
```

The Python API exposes the same blocking workflow:

```python
from firstmove_eval import load_config, run, validate

config = load_config("examples/config.mock.yaml")
report = validate(config)
if report.valid:
    run_report = run(config)
```

For an OpenAI-compatible Chat Completions server, start from
`examples/config.openai.yaml`. The base URL must end in `/v1`; FirstMove Eval sends one
`POST /chat/completions` request per example with bounded concurrency. Transient retries are
limited, but a retry after an ambiguous timeout can result in another provider charge.

## Result semantics

Successful but unparseable or illegal model answers count as incorrect in reference and
engine best-move accuracy. Provider failures are excluded from model-quality denominators
and reported through coverage and structured errors. Numeric centipawn loss is defined only
for legal predictions where both Stockfish searches return centipawn scores; mate scores are
preserved as typed values and numeric loss is skipped.

See [artifact schemas](docs/artifacts.md) and [metric semantics](docs/metrics.md) for the
versioned contracts.

## Reproducibility and privacy

Run directories contain a normalized input snapshot, per-example JSONL, error JSONL, a
manifest, and a summary. Raw prompts and responses are stored by default and can be disabled.
Credential values are read from an environment variable and never serialized. Exact engine
reproduction requires the recorded Stockfish binary, options, and comparable runtime
conditions.

## Licensing

FirstMove Eval is licensed under GPL-3.0-or-later. `python-chess` is also GPL-3.0-or-later.
Stockfish is an external GPLv3 program; if you redistribute Stockfish, comply with its own
source and license obligations.
