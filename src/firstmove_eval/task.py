"""Standard-chess first-move task normalization and interpretation."""

from __future__ import annotations

import re
from typing import Any

import chess
from pydantic import ValidationError

from firstmove_eval._json import canonical_json, sha256_text
from firstmove_eval.errors import DatasetRowError
from firstmove_eval.models import (
    ChatMessage,
    ChessInput,
    MoveInterpretation,
    NormalizedExample,
    NormalizedReference,
    SourceExample,
)

PROMPT_VERSION = "chess-uci-v1"
SYSTEM_PROMPT = (
    "You are given a standard-chess position in FEN. Return exactly one best move in "
    "UCI coordinate notation and nothing else."
)
PROMPT_TEMPLATE_HASH = sha256_text(
    canonical_json(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "FEN: {fen}"},
        ]
    )
)

_FENCE_RE = re.compile(r"^```(?:[A-Za-z0-9_-]+)?[ \t]*\r?\n(.*)\r?\n```$", re.DOTALL)


class ChessFirstMoveTask:
    """All chess-aware behavior for the V1 task."""

    name = "chess_first_move"
    version = "1"
    prompt_version = PROMPT_VERSION
    prompt_template_hash = PROMPT_TEMPLATE_HASH

    def normalize_row(self, raw: Any) -> NormalizedExample:
        try:
            source = SourceExample.model_validate(raw)
        except ValidationError as exc:
            raise DatasetRowError("invalid_schema", _compact_validation_error(exc)) from exc

        fen = source.input.fen.strip()
        if len(fen.split()) != 6:
            raise DatasetRowError("invalid_fen", "FEN must contain exactly six fields")
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise DatasetRowError("invalid_fen", f"FEN cannot be parsed: {exc}") from exc
        if board.status() != chess.STATUS_VALID:
            raise DatasetRowError(
                "invalid_position",
                f"FEN fails structural validity checks (status={int(board.status())})",
            )
        if board.is_game_over() or not any(board.legal_moves):
            raise DatasetRowError(
                "terminal_position",
                "position has no legal move and cannot be used for a first-move task",
            )

        normalized_reference: NormalizedReference | None = None
        if source.reference is not None:
            notation = source.reference.move.strip()
            move = _parse_legal_reference(board, notation)
            if move is None:
                raise DatasetRowError(
                    "invalid_reference",
                    "reference move is not one legal UCI or SAN move in the supplied FEN",
                )
            normalized_reference = NormalizedReference(
                uci=move.uci(),
                source_notation=notation,
                source_line=source.reference.source_line,
            )

        return NormalizedExample(
            id=source.id,
            input=ChessInput(fen=board.fen(en_passant="fen"), variant="standard"),
            reference=normalized_reference,
            metadata=source.metadata,
        )

    def build_messages(self, example: NormalizedExample) -> tuple[ChatMessage, ...]:
        return (
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=f"FEN: {example.input.fen}"),
        )

    def interpret(self, example: NormalizedExample, text: str) -> MoveInterpretation:
        board = chess.Board(example.input.fen)
        forced_move = board.legal_moves.count() == 1
        stripped = text.strip()
        if not stripped:
            return MoveInterpretation(
                parse_success=False,
                legal=False,
                predicted_uci=None,
                notation=None,
                candidate=None,
                diagnostic="empty_response",
                format_compliant=False,
                extra_text=False,
                forced_move=forced_move,
            )

        wrapper_used = False
        fence_match = _FENCE_RE.fullmatch(stripped)
        if fence_match:
            wrapper_used = True
            stripped = fence_match.group(1).strip()

        nonempty_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if not nonempty_lines:
            return MoveInterpretation(
                parse_success=False,
                legal=False,
                predicted_uci=None,
                notation=None,
                candidate=None,
                diagnostic="empty_response",
                format_compliant=False,
                extra_text=wrapper_used,
                forced_move=forced_move,
            )

        candidate = nonempty_lines[0]
        extra_text = wrapper_used or len(nonempty_lines) > 1

        try:
            move = chess.Move.from_uci(candidate)
        except ValueError:
            move = None
        if move is not None:
            legal = move in board.legal_moves
            return MoveInterpretation(
                parse_success=True,
                legal=legal,
                predicted_uci=move.uci(),
                notation="uci",
                candidate=candidate,
                diagnostic="ok" if legal else "illegal_move",
                format_compliant=not extra_text,
                extra_text=extra_text,
                forced_move=forced_move,
            )

        try:
            san_move = board.parse_san(candidate)
        except ValueError:
            return MoveInterpretation(
                parse_success=False,
                legal=False,
                predicted_uci=None,
                notation=None,
                candidate=candidate,
                diagnostic="invalid_or_ambiguous_move",
                format_compliant=False,
                extra_text=extra_text,
                forced_move=forced_move,
            )
        return MoveInterpretation(
            parse_success=True,
            legal=True,
            predicted_uci=san_move.uci(),
            notation="san",
            candidate=candidate,
            diagnostic="ok",
            format_compliant=False,
            extra_text=extra_text,
            forced_move=forced_move,
        )


def _parse_legal_reference(board: chess.Board, notation: str) -> chess.Move | None:
    try:
        move = chess.Move.from_uci(notation)
    except ValueError:
        move = None
    if move is not None:
        return move if move in board.legal_moves else None
    try:
        return board.parse_san(notation)
    except ValueError:
        return None


def _compact_validation_error(error: ValidationError) -> str:
    parts: list[str] = []
    for item in error.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(piece) for piece in item["loc"])
        parts.append(f"{location}: {item['msg']}" if location else str(item["msg"]))
    return "; ".join(parts)
