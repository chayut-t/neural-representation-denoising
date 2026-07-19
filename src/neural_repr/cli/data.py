"""``neural-repr-data`` — dataset download, validation, and manifests.

Phase 2 scaffold: the app and its command surface exist so CLI wiring and
``--help`` work; dataset logic arrives in Phase 4 (plan §7).
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="neural-repr-data",
    help="Download, validate, and manifest public datasets (DIV2K, BSDS500).",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def download(
    dataset: str = typer.Argument(..., help="Dataset name, e.g. 'div2k' or 'bsds500'."),
    accept_terms: bool = typer.Option(
        False,
        "--accept-terms",
        help="Explicitly accept the dataset's license terms before download.",
    ),
) -> None:
    """Download a public dataset from its official source (Phase 4)."""
    raise typer.Exit(_not_yet(f"download {dataset!r} (accept_terms={accept_terms})", phase=4))


@app.command()
def check() -> None:
    """Validate local datasets, hashes, and split manifests (Phase 4)."""
    raise typer.Exit(_not_yet("data-check", phase=4))


def _not_yet(what: str, *, phase: int) -> int:
    # Return non-zero: an invoked operation that did no work must fail (codex P2).
    typer.echo(
        f"[neural-repr-data] '{what}' is not implemented yet (planned: Phase {phase}).", err=True
    )
    return 1


def main() -> None:
    app()


if __name__ == "__main__":
    main()
