"""``neural-repr-eval`` — evaluate trained models and write metrics.

Phase 2 scaffold: CLI wiring and ``--help`` work; evaluation logic arrives with
the study implementations (plan §7 Phases 6-10).
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="neural-repr-eval",
    help="Evaluate trained models and write validated metric records.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def run(
    run_id: str = typer.Argument(..., help="Run ID to evaluate (under runs/…)."),
) -> None:
    """Evaluate a completed run and emit its metric records (Phases 6-10)."""
    raise typer.Exit(_not_yet(f"eval {run_id!r}", phase="6-10"))


def _not_yet(what: str, *, phase: object) -> int:
    # Return non-zero: an invoked operation that did no work must fail (codex P2).
    typer.echo(
        f"[neural-repr-eval] '{what}' is not implemented yet (planned: Phase {phase}).", err=True
    )
    return 1


def main() -> None:
    app()


if __name__ == "__main__":
    main()
