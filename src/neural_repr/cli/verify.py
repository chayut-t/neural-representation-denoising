"""``neural-repr-verify`` — environment info and release verification.

The ``system-info`` command (plan §7 Phase 2, task 8) writes the sanitized
execution-environment record (§6.1) with two-layer provenance fields. Other
verification subcommands (release checks, baseline hashes) arrive later.
"""

from __future__ import annotations

import json
import pathlib

import typer

from neural_repr.provenance import (
    collect_execution_environment,
    validate_execution_record,
)

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


@app.command("check-image")
def check_image() -> None:
    """Self-check for the public reference container (Gate P2 image checks).

    Runs, inside whatever environment invokes it: package import, a
    provenance-schema validation of the ``system-info`` record, and a
    deterministic numerical smoke computation. Exits non-zero on any failure so
    it can gate the built public image in CI.
    """
    import neural_repr  # import check
    from neural_repr.common import numerical_smoke
    from neural_repr.provenance.leak_patterns import canaries_all_detected

    record = collect_execution_environment(install_mode="public-reference-container").to_record()
    validate_execution_record(record)  # provenance-schema check (raises on violation)

    smoke = numerical_smoke(seed=0)
    repeat = numerical_smoke(seed=0)
    if smoke["fingerprint"] != repeat["fingerprint"]:
        typer.echo("numerical smoke is not repeatable within process", err=True)
        raise typer.Exit(1)

    if not canaries_all_detected():
        typer.echo("synthetic-canary leak scan failed to flag known patterns", err=True)
        raise typer.Exit(1)

    typer.echo(
        f"check-image OK: neural_repr {neural_repr.__version__}; "
        f"provenance-schema valid; numerical-smoke {smoke['fingerprint']}; canary-scan armed"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
