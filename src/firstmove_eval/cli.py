"""FirstMove Eval command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from firstmove_eval.config import load_config
from firstmove_eval.errors import FirstMoveEvalError
from firstmove_eval.runner import run as run_evaluation
from firstmove_eval.runner import validate as validate_evaluation

app = typer.Typer(
    name="firstmove-eval",
    help="Evaluate one-move standard-chess answers with auditable artifacts.",
    no_args_is_help=True,
)


@app.command("validate")
def validate_command(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
) -> None:
    """Validate config, dataset, credentials, and Stockfish without model calls."""

    try:
        report = validate_evaluation(load_config(config_path))
    except (FirstMoveEvalError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    if not report.valid:
        raise typer.Exit(code=1)


@app.command("run")
def run_command(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
) -> None:
    """Run an evaluation and print its terminal report."""

    try:
        report = run_evaluation(load_config(config_path))
    except KeyboardInterrupt as exc:
        raise typer.Exit(code=130) from exc
    except (FirstMoveEvalError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    if report.exit_code:
        raise typer.Exit(code=report.exit_code)
