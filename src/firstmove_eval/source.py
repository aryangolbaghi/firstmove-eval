"""Bounded-memory JSONL preflight and canonical snapshot creation."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from firstmove_eval._json import canonical_json
from firstmove_eval.errors import DatasetRowError
from firstmove_eval.models import NormalizedExample, ValidationIssue
from firstmove_eval.task import ChessFirstMoveTask


@dataclass(frozen=True, slots=True)
class PreflightResult:
    valid: bool
    example_count: int
    issue_count: int
    issues: tuple[ValidationIssue, ...]
    raw_sha256: str
    normalized_sha256: str | None


class JsonlSource:
    """Decode source records without containing any domain logic."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def preflight(
        self,
        task: ChessFirstMoveTask,
        snapshot_path: Path,
        on_issue: Callable[[ValidationIssue], None] | None = None,
        issue_sample_limit: int = 1_000,
    ) -> PreflightResult:
        temporary = snapshot_path.with_name(f".{snapshot_path.name}.tmp")
        id_database = snapshot_path.with_name(f".{snapshot_path.name}.ids.sqlite3")
        raw_digest = hashlib.sha256()
        normalized_digest = hashlib.sha256()
        sampled_issues: list[ValidationIssue] = []
        issue_count = 0
        example_count = 0

        def report(issue: ValidationIssue) -> None:
            nonlocal issue_count
            issue_count += 1
            if len(sampled_issues) < issue_sample_limit:
                sampled_issues.append(issue)
            if on_issue is not None:
                on_issue(issue)

        id_database.unlink(missing_ok=True)
        connection = sqlite3.connect(id_database)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("CREATE TABLE ids (id TEXT PRIMARY KEY) WITHOUT ROWID")
            with self.path.open("rb") as source, temporary.open("wb") as snapshot:
                for line_number, raw_line in enumerate(source, start=1):
                    raw_digest.update(raw_line)
                    if not raw_line.strip():
                        continue
                    try:
                        decoded = raw_line.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        report(
                            ValidationIssue(
                                code="invalid_utf8",
                                message=f"line is not UTF-8: {exc}",
                                line=line_number,
                            )
                        )
                        continue
                    try:
                        row: Any = json.loads(decoded)
                    except json.JSONDecodeError as exc:
                        report(
                            ValidationIssue(
                                code="invalid_json",
                                message=f"invalid JSON: {exc.msg}",
                                line=line_number,
                            )
                        )
                        continue
                    try:
                        example = task.normalize_row(row)
                    except DatasetRowError as exc:
                        report(
                            ValidationIssue(code=exc.code, message=exc.message, line=line_number)
                        )
                        continue
                    try:
                        connection.execute("INSERT INTO ids (id) VALUES (?)", (example.id,))
                    except sqlite3.IntegrityError:
                        report(
                            ValidationIssue(
                                code="duplicate_id",
                                message=f"duplicate example id: {example.id}",
                                line=line_number,
                            )
                        )
                        continue
                    encoded = (canonical_json(example.model_dump(mode="json")) + "\n").encode(
                        "utf-8"
                    )
                    snapshot.write(encoded)
                    normalized_digest.update(encoded)
                    example_count += 1
                snapshot.flush()
                os.fsync(snapshot.fileno())
            connection.commit()
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            connection.close()
            id_database.unlink(missing_ok=True)

        if example_count == 0 and issue_count == 0:
            report(
                ValidationIssue(
                    code="empty_dataset",
                    message="dataset contains no nonblank valid examples",
                )
            )

        if issue_count:
            temporary.unlink(missing_ok=True)
            normalized_hash: str | None = None
        else:
            os.replace(temporary, snapshot_path)
            normalized_hash = normalized_digest.hexdigest()

        return PreflightResult(
            valid=issue_count == 0,
            example_count=example_count,
            issue_count=issue_count,
            issues=tuple(sampled_issues),
            raw_sha256=raw_digest.hexdigest(),
            normalized_sha256=normalized_hash,
        )

    @staticmethod
    def read_snapshot(path: Path) -> Iterator[NormalizedExample]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield NormalizedExample.model_validate_json(line)
