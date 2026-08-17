# Artifact schema V1

Every run uses a unique directory. JSON objects use `schema_version: 1`; readers must reject
unknown major schema versions rather than guessing.

## `manifest.json`

The manifest is replaced atomically at state transitions. It records the safe resolved
configuration, package/task/prompt and metric-definition versions, dataset fingerprints,
runtime and dependency versions, optional Stockfish identity, timestamps, terminal status,
and counts. Credential values and arbitrary provider payloads are never included.

Terminal statuses are `complete`, `partial`, `interrupted`, `validation_failed`, and `failed`.
Incorrect or malformed model answers do not make a run `partial`; operational provider or
engine failures do.

## `input_snapshot.jsonl`

This is promoted atomically only after every source row passes preflight. Each row contains a
canonical FEN and canonical UCI reference when a reference was supplied. The manifest's
normalized SHA-256 is computed over the exact UTF-8 bytes in this file.

## `examples.jsonl`

Each processed snapshot row has one terminal result containing:

- input, normalized reference, and metadata;
- prompt/template hashes and optionally the complete prompt;
- one correlated generation success or error;
- optionally the raw response, always with a response hash on generation success;
- deterministic move interpretation;
- typed engine assessment when available;
- one explicit result for every configured metric.

Generation and engine errors reference an `error_id` from `errors.jsonl`.

## `errors.jsonl`

Diagnostics are serialized once with a stable code, category, stage, safe message, and
available line/example/request correlation. HTTP response bodies, headers, exception text
that may contain credentials, and credential values are not copied into this file.

## `summary.json`

For every metric, the summary reports `measured`, `skipped`, `error`, coverage, reasons, and
the declared aggregation. Boolean metrics expose numerator/rate; centipawn loss exposes
count, sum, mean, minimum, and maximum. It also reports generation coverage, forced moves,
operational-error events, and the jointly measured reference/engine agreement matrix.

