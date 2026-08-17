"""Versioned, incrementally flushed run artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TextIO

from firstmove_eval._json import canonical_json
from firstmove_eval.errors import ArtifactError
from firstmove_eval.models import ErrorRecord


class ArtifactStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            self._examples = (run_dir / "examples.jsonl").open("x", encoding="utf-8", newline="\n")
            self._errors = (run_dir / "errors.jsonl").open("x", encoding="utf-8", newline="\n")
        except OSError as exc:
            raise ArtifactError(f"cannot create run directory {run_dir}: {exc}") from exc

    @property
    def snapshot_path(self) -> Path:
        return self.run_dir / "input_snapshot.jsonl"

    def write_manifest(self, value: dict[str, Any]) -> None:
        self._atomic_json(self.run_dir / "manifest.json", value)

    def write_summary(self, value: dict[str, Any]) -> None:
        self._atomic_json(self.run_dir / "summary.json", value)

    def write_example(self, value: dict[str, Any]) -> None:
        self._write_line(self._examples, value)

    def write_error(self, value: ErrorRecord) -> None:
        self._write_line(self._errors, value.model_dump(mode="json"))

    def flush_batch(self) -> None:
        try:
            for handle in (self._examples, self._errors):
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise ArtifactError(f"cannot flush run artifacts: {exc}") from exc

    def close(self) -> None:
        for handle in (self._examples, self._errors):
            if not handle.closed:
                handle.close()

    def _atomic_json(self, path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise ArtifactError(f"cannot write {path.name}: {exc}") from exc

    @staticmethod
    def _write_line(handle: TextIO, value: dict[str, Any]) -> None:
        try:
            handle.write(canonical_json(value))
            handle.write("\n")
        except OSError as exc:
            raise ArtifactError(f"cannot write {Path(handle.name).name}: {exc}") from exc
