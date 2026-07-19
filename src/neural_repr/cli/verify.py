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
    compat_status: str = typer.Option(
        "unknown",
        "--compat-status",
        help="reference_compatibility_status category: public-reference-run | "
        "equivalence-verified | pending-equivalence | unknown.",
    ),
) -> None:
    """Print/write the sanitized execution-environment record (§2.4 layer 1, §6.1).

    Contains only vendor-neutral scientific provenance and an opaque fingerprint;
    never hostnames, registries, paths, or account IDs. The emitted record is
    validated against the closed schema before output.
    """
    env = collect_execution_environment(
        determinism_mode=determinism_mode,
        install_mode=install_mode,
        reference_compatibility_status=compat_status,
        public_reference_environment_digest=reference_digest,
    )
    record = env.to_record()
    validate_execution_record(record)  # never emit a non-schema-valid record
    payload = json.dumps(record, indent=2, default=str, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
        typer.echo(f"wrote execution-environment record to {output}")
    else:
        typer.echo(payload)


@app.command("check-image")
def check_image(
    require_cuda: bool = typer.Option(
        False,
        "--require-cuda",
        help="Fail unless CUDA is available and run the numerical smoke on the GPU.",
    ),
    expect_torch: str | None = typer.Option(
        None, "--expect-torch", help="Assert torch.__version__ starts with this value."
    ),
    expect_cuda: str | None = typer.Option(
        None,
        "--expect-cuda",
        help="Assert torch.version.cuda starts with this value (e.g. '13.0').",
    ),
    expect_python: str | None = typer.Option(
        None,
        "--expect-python",
        help="Assert the running interpreter is exactly this version (e.g. '3.11.15').",
    ),
) -> None:
    """Self-check for the public reference container (Gate P2 image checks).

    Runs, inside whatever environment invokes it: package import, a
    provenance-schema validation of the ``system-info`` record, a deterministic
    numerical smoke computation (on GPU when ``--require-cuda``), an optional
    torch/CUDA version assertion, and the synthetic-canary leak-scan self-test.
    Exits non-zero on any failure so it can gate the built public image in CI.
    """
    import platform

    import torch

    import neural_repr  # import check
    from neural_repr.common import numerical_smoke
    from neural_repr.provenance.leak_patterns import canaries_all_detected

    record = collect_execution_environment(install_mode="public-reference-container").to_record()
    validate_execution_record(record)  # provenance-schema check (raises on violation)

    # Python version assertion (A3): the reference interpreter is pinned to an exact
    # patch in .python-version; confirm the running one matches exactly.
    running_python = platform.python_version()
    if expect_python is not None and running_python != expect_python:
        typer.echo(
            f"python version mismatch: got {running_python}, expected exactly {expect_python}",
            err=True,
        )
        raise typer.Exit(1)

    # Version assertions (P1.1): confirm the running stack matches the declared one.
    if expect_torch is not None and not torch.__version__.startswith(expect_torch):
        typer.echo(
            f"torch version mismatch: got {torch.__version__}, expected {expect_torch}*", err=True
        )
        raise typer.Exit(1)
    torch_cuda = getattr(torch.version, "cuda", None)
    if expect_cuda is not None and (torch_cuda is None or not torch_cuda.startswith(expect_cuda)):
        typer.echo(f"torch CUDA mismatch: got {torch_cuda}, expected {expect_cuda}*", err=True)
        raise typer.Exit(1)

    # CUDA runtime check (P1.2): a --require-cuda run must actually exercise the GPU.
    device = "cpu"
    if require_cuda:
        if not torch.cuda.is_available():
            typer.echo("--require-cuda set but torch.cuda.is_available() is False", err=True)
            raise typer.Exit(1)
        device = "cuda"

    smoke = numerical_smoke(seed=0, device=device)
    repeat = numerical_smoke(seed=0, device=device)
    if smoke["fingerprint"] != repeat["fingerprint"]:
        typer.echo("numerical smoke is not repeatable within process", err=True)
        raise typer.Exit(1)
    if require_cuda and smoke["device"] != "cuda":
        typer.echo("numerical smoke did not run on CUDA despite --require-cuda", err=True)
        raise typer.Exit(1)

    if not canaries_all_detected():
        typer.echo("synthetic-canary leak scan failed to flag known patterns", err=True)
        raise typer.Exit(1)

    typer.echo(
        f"check-image OK: neural_repr {neural_repr.__version__}; python {running_python}; "
        f"torch {torch.__version__} (cuda {torch_cuda}); provenance-schema valid; "
        f"numerical-smoke {smoke['fingerprint']} on {smoke['device']}; canary-scan armed"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
