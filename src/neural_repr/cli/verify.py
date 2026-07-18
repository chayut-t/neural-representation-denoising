"""``neural-repr-verify`` — environment info and release verification.

The ``system-info`` command (plan §7 Phase 2, task 8) writes the sanitized
execution-environment record (§6.1) with two-layer provenance fields. Other
verification subcommands (release checks, baseline hashes) arrive later.
"""

from __future__ import annotations

import json
import pathlib

import typer

from neural_repr.provenance import collect_execution_environment

app = typer.Typer(
    name="neural-repr-verify",
    help="Environment info and release verification.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _root() -> None:
    """Environment info and release verification for neural_repr."""
    # Forces Typer to expose subcommands (system-info, plus later verify checks)
    # even while only one exists.


@app.command("system-info")
def system_info(
    output: pathlib.Path | None = typer.Option(
        None, "--output", "-o", help="Write the JSON record here (default: stdout)."
    ),
    determinism_mode: str = typer.Option(
        "unset", help="Determinism mode label: debug | research | benchmark | unset."
    ),
    install_mode: str = typer.Option(
        "unknown",
        help="How the env was materialized (category, not identifier): "
        "public-reference-container | private-mirror | local | unknown.",
    ),
    reference_digest: str | None = typer.Option(
        None,
        "--reference-digest",
        help="Digest of the publicly pullable reference container, if running under it.",
    ),
) -> None:
    """Print/write the sanitized execution-environment record (§2.4 layer 1, §6.1).

    Contains only vendor-neutral scientific provenance and an opaque
    fingerprint; never hostnames, registries, paths, or account IDs.
    """
    env = collect_execution_environment(
        determinism_mode=determinism_mode,
        install_mode=install_mode,
        public_reference_environment_digest=reference_digest,
    )
    payload = json.dumps(env.to_record(), indent=2, default=str, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
        typer.echo(f"wrote execution-environment record to {output}")
    else:
        typer.echo(payload)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
