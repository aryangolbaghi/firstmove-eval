from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import chess
import chess.engine
import pytest

from firstmove_eval.config import EngineSettings
from firstmove_eval.engine import ChessEngineService, _score_from_info
from firstmove_eval.models import EngineAssessment, EngineScore


def test_score_normalization_preserves_cp_and_mate_types() -> None:
    cp = _score_from_info(
        {"score": chess.engine.PovScore(chess.engine.Cp(42), chess.WHITE)}, chess.WHITE
    )
    assert cp.kind == "cp"
    assert cp.value == 42
    mate = _score_from_info(
        {"score": chess.engine.PovScore(chess.engine.Mate(3), chess.BLACK)},
        chess.BLACK,
    )
    assert mate.kind == "mate"
    assert mate.value == 3


def test_forced_move_does_not_start_engine() -> None:
    service = ChessEngineService(EngineSettings(enabled=True, path="unused"))
    result = service.assess("8/8/8/8/8/8/8/k1KQ4 b - - 0 1", "a1a2")
    assert result.forced_move
    assert result.best_move_uci == "a1a2"
    assert result.best_score.value == 0


def test_engine_process_is_restarted_once(monkeypatch: pytest.MonkeyPatch) -> None:
    service = ChessEngineService(EngineSettings(enabled=True, path="unused"))
    service._engine = cast(Any, object())
    calls = 0
    starts = 0
    expected = EngineAssessment(
        forced_move=False,
        best_move_uci="e2e4",
        predicted_move_uci="e2e4",
        best_score=EngineScore(kind="cp", value=20),
        predicted_score=EngineScore(kind="cp", value=20),
    )

    def assess_once(_: chess.Board, __: chess.Move) -> EngineAssessment:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise chess.engine.EngineTerminatedError("terminated")
        return expected

    def close() -> None:
        service._engine = None

    def start() -> None:
        nonlocal starts
        starts += 1
        service._engine = cast(Any, object())

    monkeypatch.setattr(service, "_assess_once", assess_once)
    monkeypatch.setattr(service, "close", close)
    monkeypatch.setattr(service, "start", start)
    result = service.assess(chess.STARTING_FEN, "e2e4")
    assert result.model_copy(update={"latency_ms": 0}) == expected
    assert result.latency_ms > 0
    assert calls == 2
    assert starts == 1


@pytest.mark.stockfish
@pytest.mark.skipif(not os.environ.get("STOCKFISH_PATH"), reason="STOCKFISH_PATH is not set")
def test_real_stockfish_root_restricted_analysis() -> None:
    path = Path(os.environ["STOCKFISH_PATH"])
    service = ChessEngineService(EngineSettings(enabled=True, path=path, nodes=1_000, hash_mb=16))
    try:
        service.start()
        result = service.assess(chess.STARTING_FEN, "e2e4")
    finally:
        service.close()
    assert result.predicted_move_uci == "e2e4"
    assert result.best_score.kind in {"cp", "mate"}
