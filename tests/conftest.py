from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from firstmove_eval.config import (
    ArtifactSettings,
    DatasetSettings,
    EngineSettings,
    MetricsSettings,
    MockModelSettings,
    RunConfig,
    RunSettings,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def source_row(
    *,
    example_id: str = "example-1",
    fen: str = START_FEN,
    reference: str | None = "e4",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": example_id,
        "input": {"fen": fen, "variant": "standard"},
        "reference": (
            {"move": reference, "source_line": [reference]} if reference is not None else None
        ),
        "metadata": {"source": "test", "split": "test"},
    }


def write_rows(path: Path, rows: list[Any]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture
def mock_config(tmp_path: Path) -> RunConfig:
    dataset = tmp_path / "positions.jsonl"
    write_rows(dataset, [source_row()])
    return RunConfig(
        run=RunSettings(output_dir=tmp_path / "runs"),
        dataset=DatasetSettings(path=dataset),
        model=MockModelSettings(default_response="e2e4"),
        metrics=MetricsSettings(engine=EngineSettings(enabled=False)),
        artifacts=ArtifactSettings(store_prompts=True, store_raw_responses=True),
    )
