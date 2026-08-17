# Release checklist

1. Reconfirm ownership and availability of the `firstmove-eval` PyPI project.
2. Run formatting, lint, strict typing, unit/contract tests, and the 100,000-row benchmark.
3. Run the opt-in real Stockfish test with a pinned official binary and record its digest.
4. Build both wheel and source distribution; run `twine check` and inspect their file lists.
5. Install the wheel in a clean Python 3.13 environment and run the documented mock example.
6. Run dependency vulnerability and license audits; update third-party notices when needed.
7. Confirm the version in `pyproject.toml`, `_version.py`, tag, and changelog agrees.
8. Publish to TestPyPI and repeat the clean-install smoke test.
9. Approve the protected `pypi` GitHub environment to publish through OIDC trusted publishing.
10. Verify provenance, wheel, sdist, license, console entry point, and README on PyPI.

