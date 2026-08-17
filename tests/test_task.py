from __future__ import annotations

import chess
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from firstmove_eval.errors import DatasetRowError
from firstmove_eval.task import ChessFirstMoveTask

from .conftest import START_FEN, source_row


def test_normalizes_san_reference_to_uci() -> None:
    example = ChessFirstMoveTask().normalize_row(source_row(reference="e4"))
    assert example.input.fen == START_FEN
    assert example.reference is not None
    assert example.reference.uci == "e2e4"
    assert example.reference.source_notation == "e4"


@pytest.mark.parametrize(
    ("row", "code"),
    [
        ({"schema_version": 1}, "invalid_schema"),
        (source_row(example_id="   "), "invalid_schema"),
        (source_row(fen="not a fen"), "invalid_fen"),
        (
            source_row(fen="7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
            "terminal_position",
        ),
        (source_row(reference="e5"), "invalid_reference"),
    ],
)
def test_rejects_invalid_rows(row: object, code: str) -> None:
    with pytest.raises(DatasetRowError) as captured:
        ChessFirstMoveTask().normalize_row(row)
    assert captured.value.code == code


@pytest.mark.parametrize(
    ("text", "parsed", "legal", "uci", "notation", "compliant", "extra"),
    [
        ("e2e4", True, True, "e2e4", "uci", True, False),
        ("e2e5", True, False, "e2e5", "uci", True, False),
        ("e4", True, True, "e2e4", "san", False, False),
        ("e2e4\nBecause it controls the centre", True, True, "e2e4", "uci", False, True),
        ("```\ne2e4\n```", True, True, "e2e4", "uci", False, True),
        ("e2e4 e7e5", False, False, None, None, False, False),
        ("", False, False, None, None, False, False),
    ],
)
def test_interpreter_contract(
    text: str,
    parsed: bool,
    legal: bool,
    uci: str | None,
    notation: str | None,
    compliant: bool,
    extra: bool,
) -> None:
    task = ChessFirstMoveTask()
    example = task.normalize_row(source_row())
    result = task.interpret(example, text)
    assert (result.parse_success, result.legal, result.predicted_uci) == (
        parsed,
        legal,
        uci,
    )
    assert result.notation == notation
    assert result.format_compliant is compliant
    assert result.extra_text is extra


def test_ambiguous_san_is_never_guessed() -> None:
    task = ChessFirstMoveTask()
    example = task.normalize_row(
        source_row(
            fen="4k3/8/8/8/8/8/8/1N2KN2 w - - 0 1",
            reference=None,
        )
    )
    result = task.interpret(example, "Nd2")
    assert not result.parse_success
    assert result.diagnostic == "invalid_or_ambiguous_move"


@settings(max_examples=50, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=255), min_size=1, max_size=40))
def test_legal_move_san_and_uci_round_trip(indices: list[int]) -> None:
    board = chess.Board()
    for index in indices:
        legal = list(board.legal_moves)
        if not legal:
            break
        move = legal[index % len(legal)]
        san = board.san(move)
        assert board.parse_san(san) == move
        assert chess.Move.from_uci(move.uci()) == move
        board.push(move)
