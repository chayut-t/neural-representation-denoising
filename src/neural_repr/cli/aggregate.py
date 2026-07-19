"""``neural-repr-aggregate`` — aggregate validated runs into statistics.

Phase 2 scaffold: CLI wiring and ``--help`` work; aggregation logic arrives in
plan §7 Phase 10.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="neural-repr-aggregate",
    help="Aggregate validated runs into paired statistics and robustness reports.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def run(
    manifest: str = typer.Argument(..., help="Aggregate manifest of validated run IDs."),
) -> None:
    """Aggregate validated runs listed in a committed manifest (Phase 10)."""
    raise typer.Exit(_not_yet(f"aggregate {manifest!r}", phase=10))


def _not_yet(what: str, *, phase: int) -> int:
    # Return non-zero: an invoked operation that did no work must fail (codex P2).
    typer.echo(
        f"[neural-repr-aggregate] '{what}' is not implemented yet (planned: Phase {phase}).",
        err=True,
    )
    return 1


def main() -> None:
    app()


if __name__ == "__main__":
    main()
