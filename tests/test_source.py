from __future__ import annotations

import json
from pathlib import Path

from firstmove_eval.source import JsonlSource
from firstmove_eval.task import ChessFirstMoveTask

from .conftest import source_row


def test_preflight_collects_errors_and_does_not_promote_snapshot(tmp_path: Path) -> None:
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text(
        "{broken\n"
        + json.dumps(source_row(example_id="duplicate"))
        + "\n"
        + json.dumps(source_row(example_id="duplicate"))
        + "\n",
        encoding="utf-8",
    )
    snapshot = tmp_path / "snapshot.jsonl"
    result = JsonlSource(dataset).preflight(ChessFirstMoveTask(), snapshot)
    assert not result.valid
    assert result.issue_count == 2
    assert {issue.code for issue in result.issues} == {"invalid_json", "duplicate_id"}
    assert not snapshot.exists()
    assert not list(tmp_path.glob("*.sqlite3"))


def test_preflight_snapshot_is_canonical_and_readable(tmp_path: Path) -> None:
    dataset = tmp_path / "good.jsonl"
    dataset.write_text("\n" + json.dumps(source_row()) + "\n", encoding="utf-8")
    snapshot = tmp_path / "snapshot.jsonl"
    result = JsonlSource(dataset).preflight(ChessFirstMoveTask(), snapshot)
    assert result.valid
    assert result.example_count == 1
    assert result.normalized_sha256
    examples = list(JsonlSource.read_snapshot(snapshot))
    assert examples[0].reference is not None
    assert examples[0].reference.uci == "e2e4"
    assert not list(tmp_path.glob("*.sqlite3"))


def test_empty_dataset_is_invalid(tmp_path: Path) -> None:
    dataset = tmp_path / "empty.jsonl"
    dataset.write_text("\n\n", encoding="utf-8")
    result = JsonlSource(dataset).preflight(ChessFirstMoveTask(), tmp_path / "snapshot.jsonl")
    assert not result.valid
    assert result.issues[0].code == "empty_dataset"
