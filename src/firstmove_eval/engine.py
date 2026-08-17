"""Managed Stockfish analysis for chess-domain metrics."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

import chess
import chess.engine
from pydantic import JsonValue

from firstmove_eval.config import EngineSettings
from firstmove_eval.errors import EngineEvaluationError
from firstmove_eval.models import EngineAssessment, EngineScore


class ChessEngineService:
    """Own one UCI engine process and one bounded restart per assessment."""

    def __init__(self, settings: EngineSettings) -> None:
        if settings.path is None:
            raise ValueError("engine path is required")
        self.settings = settings
        self.path = Path(settings.path)
        self._engine: chess.engine.SimpleEngine | None = None
        self._identity: dict[str, JsonValue] | None = None

    @property
    def identity(self) -> dict[str, JsonValue]:
        if self._identity is None:
            raise RuntimeError("engine has not been started")
        return dict(self._identity)

    def start(self) -> None:
        self.close()
        engine: chess.engine.SimpleEngine | None = None
        try:
            engine = chess.engine.SimpleEngine.popen_uci(
                str(self.path), timeout=float(self.settings.timeout_seconds)
            )
            requested: dict[str, Any] = {
                "Threads": 1,
                "Hash": self.settings.hash_mb,
                "UCI_LimitStrength": False,
            }
            supported = {key: value for key, value in requested.items() if key in engine.options}
            if supported:
                engine.configure(supported)
            if "SyzygyPath" in engine.options:
                engine.configure({"SyzygyPath": "<empty>"})
        except Exception:
            if engine is not None:
                with suppress(Exception):
                    engine.quit()
            raise
        self._engine = engine
        self._identity = {
            "name": str(engine.id.get("name", "unknown")),
            "author": str(engine.id.get("author", "unknown")),
            "executable_sha256": _sha256_file(self.path),
            "threads": 1,
            "hash_mb": self.settings.hash_mb,
            "nodes": self.settings.nodes,
            "timeout_seconds": float(self.settings.timeout_seconds),
            "tablebases": "disabled",
            "ponder": False,
            "uci_limit_strength": False,
            "clear_hash_between_searches": True,
        }

    def close(self) -> None:
        engine, self._engine = self._engine, None
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                with suppress(Exception):
                    engine.close()

    def assess(self, fen: str, predicted_uci: str) -> EngineAssessment:
        started = time.perf_counter()
        board = chess.Board(fen)
        predicted_move = chess.Move.from_uci(predicted_uci)
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 1:
            only_move = legal_moves[0]
            if predicted_move != only_move:
                raise EngineEvaluationError("predicted move is not the forced legal move")
            zero = EngineScore(kind="cp", value=0)
            return EngineAssessment(
                forced_move=True,
                best_move_uci=only_move.uci(),
                predicted_move_uci=predicted_move.uci(),
                best_score=zero,
                predicted_score=zero,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        first_error: Exception | None = None
        for attempt in range(2):
            try:
                if self._engine is None:
                    self.start()
                result = self._assess_once(board, predicted_move)
                return result.model_copy(
                    update={"latency_ms": (time.perf_counter() - started) * 1000}
                )
            except (
                chess.engine.EngineError,
                chess.engine.EngineTerminatedError,
                TimeoutError,
                OSError,
            ) as exc:
                first_error = exc
                self.close()
                if attempt == 0:
                    continue
        raise EngineEvaluationError(
            f"Stockfish analysis failed after restart: {type(first_error).__name__}"
        ) from first_error

    def _assess_once(self, board: chess.Board, predicted_move: chess.Move) -> EngineAssessment:
        if self._engine is None:
            raise RuntimeError("engine is not started")
        limit = chess.engine.Limit(nodes=self.settings.nodes)

        self._clear_hash()
        best_info = self._engine.analyse(
            board,
            limit,
            info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
        )
        best_pv = best_info.get("pv")
        if not isinstance(best_pv, list) or not best_pv:
            raise chess.engine.EngineError("Stockfish returned no principal variation")
        best_move = best_pv[0]
        best_score = _score_from_info(best_info, board.turn)

        self._clear_hash()
        predicted_info = self._engine.analyse(
            board,
            limit,
            root_moves=[predicted_move],
            info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
        )
        predicted_score = _score_from_info(predicted_info, board.turn)
        clamped = (
            best_score.kind == "cp"
            and predicted_score.kind == "cp"
            and best_score.value - predicted_score.value < 0
        )
        return EngineAssessment(
            forced_move=False,
            best_move_uci=best_move.uci(),
            predicted_move_uci=predicted_move.uci(),
            best_score=best_score,
            predicted_score=predicted_score,
            negative_loss_clamped=clamped,
            best_nodes=_optional_int(best_info.get("nodes")),
            predicted_nodes=_optional_int(predicted_info.get("nodes")),
            best_depth=_optional_int(best_info.get("depth")),
            predicted_depth=_optional_int(predicted_info.get("depth")),
        )

    def _clear_hash(self) -> None:
        if self._engine is not None and "Clear Hash" in self._engine.options:
            self._engine.configure({"Clear Hash": None})


def _score_from_info(info: Mapping[str, Any], turn: chess.Color) -> EngineScore:
    pov_score = info.get("score")
    if not isinstance(pov_score, chess.engine.PovScore):
        raise chess.engine.EngineError("Stockfish returned no usable score")
    score = pov_score.pov(turn)
    if score.is_mate():
        mate = score.mate()
        if mate is None:
            raise chess.engine.EngineError("Stockfish returned an invalid mate score")
        return EngineScore(kind="mate", value=mate)
    centipawns = score.score()
    if centipawns is None:
        raise chess.engine.EngineError("Stockfish returned an invalid centipawn score")
    return EngineScore(kind="cp", value=centipawns)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
