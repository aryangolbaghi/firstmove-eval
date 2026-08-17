# FirstMove Eval 1.0.0 release evidence

Date: 2026-08-18

This file records the local release-candidate checks completed before the public release.
The distributions uploaded by GitHub Actions are rebuilt from the tagged commit and have
their own PyPI provenance and digests.

## Quality gates

- Ruff formatting: passed (34 files)
- Ruff lint: passed
- mypy strict mode: passed (17 source files)
- pytest without an engine configured: 47 passed, 1 opt-in Stockfish test skipped
- dependency audit: no known vulnerabilities
- AGPL dependency policy check: passed
- wheel and source distribution: built successfully
- Twine metadata validation: passed
- clean wheel installation and documented validation example: passed

## Stockfish integration

- Release: Stockfish 18 (`sf_18`)
- Archive: `stockfish-windows-x86-64.zip`
- Archive SHA-256: `40cc975817e7eee270b03f354810d20956df565420d320f6dd37d454dc81a139`
- Executable SHA-256: `9bde420202717ce083412027fbfb8c5c935b537591d712be8a8a8bae92f6e8d6`
- Opt-in real-engine integration test: passed

Stockfish was downloaded temporarily from the official release archive and was not added
to this repository or either Python distribution.

## Scale benchmark

- Canonical input rows: 100,000
- Elapsed time: 14.80 seconds
- Peak resident memory: 47.43 MiB
- Configured failure threshold: 128 MiB

The benchmark was run on Windows with the implementation's disk-backed preflight path.

## Local candidate artifacts

- Wheel SHA-256: `ebd3705066611ec06638cc6cc8192de9492b3fa4d73424bf26f0470bfb5126cf`
- Source distribution SHA-256: `e50e5dbc4c5ac8e4b54ed410bae70a339652268ba46d12b34620461cf836b86b`

These local digests identify the locally verified candidates only. The authoritative public
release files are built by the protected GitHub release workflow from tag `v1.0.0`.
