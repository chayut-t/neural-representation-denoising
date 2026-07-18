"""``neural-repr-train`` — train sparse-coding and attractor models.

Phase 2 scaffold: CLI wiring and ``--help`` work; training logic arrives with
the study implementations (plan §7 Phases 5-9).
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="neural-repr-train",
    help="Train models for the sparse-coding and attractor studies (Hydra-configured).",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def run(
    experiment: str = typer.Argument(
        ..., help="Hydra experiment group, e.g. 'sparse/…' or 'attractor/…'."
    ),
    overrides: list[str] = typer.Argument(None, help="Hydra-style config overrides."),
) -> None:
    """Launch a training run under a resolved Hydra config (Phases 5-9)."""
    raise typer.Exit(_not_yet(f"train {experiment!r} overrides={overrides or []}", phase="5-9"))


def _not_yet(what: str, *, phase: object) -> int:
    typer.echo(f"[neural-repr-train] '{what}' is not implemented yet (planned: Phase {phase}).")
    return 0


def main() -> None:
    app()


if __name__ == "__main__":
    main()
