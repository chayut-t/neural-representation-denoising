"""``neural-repr-figure`` — generate figures and tables from result records.

Phase 2 scaffold: CLI wiring and ``--help`` work; figure/table generation
arrives in plan §7 Phase 11.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="neural-repr-figure",
    help="Generate dissertation figures and tables from validated result records.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def make(
    artifact_id: str = typer.Argument(..., help="Stable artifact ID, e.g. 'ch3-basis-atlas'."),
) -> None:
    """Generate one figure/table artifact from its committed config (Phase 11)."""
    raise typer.Exit(_not_yet(f"figure {artifact_id!r}", phase=11))


def _not_yet(what: str, *, phase: int) -> int:
    typer.echo(f"[neural-repr-figure] '{what}' is not implemented yet (planned: Phase {phase}).")
    return 0


def main() -> None:
    app()


if __name__ == "__main__":
    main()
