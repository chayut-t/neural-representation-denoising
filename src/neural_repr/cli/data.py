"""``neural-repr-data`` — dataset registry, terms, manifests, and fixtures.

Phase 4 (plan §7). Subcommands:

* ``list`` — show registered datasets and their official/terms URLs.
* ``accept-terms`` — record local acceptance of a dataset's terms (git-ignored).
* ``verify-archive`` — verify a downloaded official archive against a recorded hash.
* ``make-fixture`` — generate the deterministic synthetic RGB fixture + manifest.
* ``check`` — validate a local dataset directory against its manifest.

Downloading multi-GB third-party archives requires authenticated access to official
sources and is intentionally NOT automated here (decision 0001: download only from
official sources after explicit local terms acceptance; never mirror/commit data).
``accept-terms`` + ``verify-archive`` implement the "accept then pin then verify"
flow; ``data/README.md`` documents the operator steps.
"""

from __future__ import annotations

import pathlib

import typer

from neural_repr.data import registry
from neural_repr.data.records import (
    ArchiveHash,
    TermsAcceptance,
    verify_archive,
    write_terms_acceptance,
)

app = typer.Typer(
    name="neural-repr-data",
    help="Registry, terms, manifests, and fixtures for public datasets (DIV2K, BSDS500).",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _root() -> None:
    """Dataset registry, terms acceptance, manifests, and fixtures."""


@app.command("list")
def list_datasets() -> None:
    """List registered datasets with their official source and terms URLs."""
    for name in registry.registered_datasets():
        spec = registry.get_dataset(name)
        typer.echo(f"{spec.name}: {spec.description}")
        typer.echo(f"  official: {spec.official_url}")
        typer.echo(f"  terms:    {spec.terms_url}  (accept with {spec.terms_acceptance_token})")
        typer.echo(f"  archives: {', '.join(spec.archives)}")


@app.command("accept-terms")
def accept_terms(
    dataset: str = typer.Argument(..., help="Dataset name, e.g. 'div2k' or 'bsds500'."),
    token: str = typer.Option(
        ...,
        "--token",
        help="The dataset-specific acceptance flag (see `list`), e.g. "
        "--accept-academic-research-terms. Must match the registry token exactly.",
    ),
    terms_file: pathlib.Path = typer.Option(
        ...,
        "--terms-file",
        help="Local copy of the exact terms text you accepted; its SHA-256 is recorded "
        "so the acceptance is bound to specific terms content, not just a URL.",
    ),
    at_iso: str = typer.Option(
        ..., "--at", help="Acceptance timestamp (ISO-8601); recorded locally, not committed."
    ),
    records_dir: pathlib.Path = typer.Option(
        pathlib.Path("data/raw/records"),
        "--records-dir",
        help="Where to write the (git-ignored) acceptance record.",
    ),
) -> None:
    """Record local acceptance of a dataset's terms (git-ignored; never committed).

    Requires the dataset-specific acceptance token (the plan's
    ``--accept-academic-research-terms`` / ``--accept-noncommercial-research-terms``) and
    a local copy of the terms text, whose hash binds the acceptance to specific content.
    """
    from neural_repr.data.records import sha256_file

    spec = registry.get_dataset(dataset)
    if token != spec.terms_acceptance_token:
        typer.echo(
            f"Refusing: {spec.name} requires token {spec.terms_acceptance_token!r}, got {token!r}",
            err=True,
        )
        raise typer.Exit(1)
    if not terms_file.is_file():
        typer.echo(f"terms file not found: {terms_file}", err=True)
        raise typer.Exit(1)
    if not _valid_iso8601(at_iso):
        typer.echo(f"--at must be an ISO-8601 timestamp, got {at_iso!r}", err=True)
        raise typer.Exit(1)
    path = write_terms_acceptance(
        records_dir,
        TermsAcceptance(
            dataset=spec.name,
            terms_url=spec.terms_url,
            terms_acceptance_token=spec.terms_acceptance_token,
            terms_sha256=sha256_file(terms_file),
            accepted_at_iso=at_iso,
        ),
    )
    typer.echo(f"recorded terms acceptance for {spec.name} -> {path}")


def _valid_iso8601(value: str) -> bool:
    from datetime import datetime

    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


@app.command("verify-archive")
def verify_archive_cmd(
    dataset: str = typer.Argument(..., help="Dataset name."),
    archive: str = typer.Argument(..., help="Official archive filename (e.g. DIV2K_train_HR.zip)."),
    path: pathlib.Path = typer.Argument(..., help="Path to the downloaded archive."),
    records_dir: pathlib.Path = typer.Option(
        pathlib.Path("data/raw/records"), "--records-dir", help="Where hash records live."
    ),
    record: bool = typer.Option(
        False,
        "--record",
        help="Record this archive's hash (only from a TRUSTED first download).",
    ),
) -> None:
    """Verify a downloaded archive against its recorded hash (or record it with --record)."""
    spec = registry.get_dataset(dataset)
    if archive not in spec.archives:
        typer.echo(f"{archive!r} is not an official {spec.name} archive: {spec.archives}", err=True)
        raise typer.Exit(1)
    if not path.is_file():
        typer.echo(f"archive not found: {path}", err=True)
        raise typer.Exit(1)
    if record:
        from neural_repr.data.records import ArchivePinConflict, record_archive_hash, sha256_file

        digest = sha256_file(path)
        try:
            record_archive_hash(
                records_dir, ArchiveHash(dataset=spec.name, archive=archive, sha256=digest)
            )
        except (ArchivePinConflict, ValueError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        typer.echo(f"recorded {spec.name}/{archive} = {digest}")
        return
    try:
        verify_archive(records_dir, spec.name, archive, path)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"OK: {spec.name}/{archive} matches its recorded hash")


@app.command("make-fixture")
def make_fixture(
    out_dir: pathlib.Path = typer.Option(
        pathlib.Path("data/raw/synthetic"), "--out-dir", help="Where to write fixture images."
    ),
    manifest: pathlib.Path = typer.Option(
        pathlib.Path("data/manifests/synthetic.csv"), "--manifest", help="Manifest CSV output."
    ),
    n: int = typer.Option(8, "--n", help="Number of synthetic images."),
    size: int = typer.Option(32, "--size", help="Image side length."),
    seed: int = typer.Option(0, "--seed", help="Fixture seed (determinism)."),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing fixture files/manifest instead of failing on collision.",
    ),
) -> None:
    """Generate the deterministic synthetic RGB fixture and write its manifest.

    Collision-failing by default (plan §0.3): if the images or manifest already exist,
    the command refuses rather than clobbering. Because the fixture is deterministic,
    the intended way to confirm an existing fixture is `check` (which re-hashes), and
    intentional regeneration uses `--force`.
    """
    import io as _io

    from PIL import Image

    from neural_repr.data.io_safe import atomic_write_bytes
    from neural_repr.data.manifests import build_manifest_row, write_manifest
    from neural_repr.data.synthetic import synthetic_image

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for i in range(n):
            img = synthetic_image(i, size=size, seed=seed)
            rel = f"synth_{i:04d}.png"
            buf = _io.BytesIO()
            Image.fromarray(img, mode="RGB").save(buf, format="PNG")
            atomic_write_bytes(out_dir / rel, buf.getvalue(), overwrite=force)
            # The fixture is regenerated fresh on each platform; its contract is
            # decoded-pixel identity, not PNG-encoded-byte identity (zlib output varies
            # by platform/library). Record only the platform-stable content digest.
            rows.append(
                build_manifest_row(
                    out_dir,
                    out_dir / rel,
                    split_id=f"synth-{i:04d}",
                    role="train",
                    include_file_hash=False,
                )
            )
        write_manifest(rows, manifest, overwrite=force)
    except FileExistsError as exc:
        typer.echo(
            f"{exc}\n(use --force to regenerate, or `check` to verify the existing fixture)",
            err=True,
        )
        raise typer.Exit(1) from exc
    typer.echo(f"wrote {n} synthetic images to {out_dir} and manifest {manifest}")


# Default manifest + dataset facts per known dataset. 'synthetic' is the fixture
# (not a registry dataset); div2k/bsds500 come from the registry.
_DEFAULT_MANIFESTS: dict[str, str] = {
    "synthetic": "data/manifests/synthetic.csv",
    "div2k": "data/manifests/div2k.csv",
    "bsds500": "data/manifests/bsds500.csv",
}


@app.command("check")
def check(
    dataset: str = typer.Argument(
        "synthetic", help="Dataset name: 'synthetic', 'div2k', or 'bsds500'."
    ),
    manifest: pathlib.Path | None = typer.Option(
        None, "--manifest", help="Manifest CSV (default: the dataset's committed manifest)."
    ),
    root: pathlib.Path | None = typer.Option(
        None,
        "--root",
        help="Dataset root for on-disk verification (reopen/rehash each image). "
        "If omitted, only manifest structure is validated.",
    ),
) -> None:
    """Validate a dataset's manifest — and, with ``--root``, the actual bytes it points at.

    Structure checks (counts/IDs/roles/shapes/formats/modes/digest-syntax/path-safety/
    duplicate hashes/leakage) always run. With ``--root``, every row's file is resolved
    under the root (traversal-guarded), reopened/decoded, and its format/shape/mode/
    SHA-256 compared to the manifest (R2 finding 1). The dataset name selects its
    committed manifest and (for registry datasets) the expected roles.
    """
    from neural_repr.data.manifests import read_manifest, validate_manifest, verify_manifest_on_disk

    known = set(_DEFAULT_MANIFESTS)
    if dataset not in known:
        typer.echo(f"unknown dataset {dataset!r}; expected one of {sorted(known)}", err=True)
        raise typer.Exit(1)

    manifest_path = manifest if manifest is not None else pathlib.Path(_DEFAULT_MANIFESTS[dataset])
    if not manifest_path.is_file():
        typer.echo(f"manifest not found: {manifest_path}", err=True)
        raise typer.Exit(1)

    # Expected roles/IDs: registry datasets declare them; the synthetic fixture is all-train.
    expected_ids: set[str] | None = None
    if dataset == "synthetic":
        expected_roles: set[str] | None = {"train"}
    else:
        spec = registry.get_dataset(dataset)
        expected_roles = {s.role for s in spec.splits}
        # Enforce the exact expected split IDs when the dataset has real integer ranges
        # (DIV2K). BSDS500's test split is enumerated from the archive, so its IDs
        # cannot be predicted without the bytes — roles are still enforced.
        int_ids = spec.all_ids()
        if any(s.last_id > s.first_id for s in spec.splits) and int_ids:
            expected_ids = {spec.id_format.format(i) for i in int_ids}

    rows = read_manifest(manifest_path)
    if root is not None:
        problems = verify_manifest_on_disk(
            rows, root, expected_roles=expected_roles, expected_ids=expected_ids
        )
        scope = f"on-disk under {root}"
    else:
        problems = validate_manifest(rows)
        if expected_roles is not None:
            for extra in sorted({r.role for r in rows} - expected_roles):
                problems.append(f"role {extra!r} not an expected role {sorted(expected_roles)}")
        if expected_ids is not None:
            seen = {r.split_id for r in rows}
            for extra in sorted(seen - expected_ids):
                problems.append(f"unexpected split_id {extra!r} not in the expected set")
            if not (seen & expected_ids):
                problems.append("no manifest split_id matches this dataset's expected IDs")
        scope = "manifest-only (pass --root to verify bytes)"

    if problems:
        typer.echo(
            f"validation FAILED for {dataset} ({len(problems)} problems, {scope}):", err=True
        )
        for p in problems:
            typer.echo(f"  - {p}", err=True)
        raise typer.Exit(1)
    typer.echo(f"[data-check] {dataset}: {len(rows)} images valid ({scope})")


@app.command("audit")
def audit(
    out: pathlib.Path = typer.Option(
        pathlib.Path("data/interim/synthetic-audit.json"),
        "--out",
        help="Where to write the JSON audit report.",
    ),
    n: int = typer.Option(8, "--n", help="Number of synthetic images to audit."),
    size: int = typer.Option(48, "--size", help="Synthetic image side length."),
    seed: int = typer.Option(0, "--seed", help="Fixture seed."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing report."),
) -> None:
    """Generate a quantitative data-audit report on the synthetic fixture (plan task 11).

    Runs the representation-track transforms (opponent -> DC-removed -> train-only
    whitening) and writes the assembled audit report (histograms, patch power,
    whitening covariance, train/test similarity, before/after) as JSON.
    """
    import numpy as np

    from neural_repr.data.audit import build_audit_report, write_audit_report
    from neural_repr.data.color import remove_dc, rgb_to_opponent
    from neural_repr.data.patches import PatchConfig, sample_random_patches
    from neural_repr.data.synthetic import synthetic_image
    from neural_repr.data.whitening import fit_whitening

    cfg = PatchConfig(size=16, n_random=64)
    split = max(1, n // 2)  # first half train, second half test (disjoint by image)
    train_rgb: list[np.ndarray] = []
    test_rgb: list[np.ndarray] = []
    for i in range(n):
        img = synthetic_image(i, size=size, seed=seed).astype("float64") / 255.0
        patches = sample_random_patches(
            img, cfg, dataset_sha256="synthetic", image_id=f"{i:04d}", seed=seed
        )
        (train_rgb if i < split else test_rgb).append(patches)

    train_patches = np.concatenate(train_rgb, axis=0)
    test_patches = np.concatenate(test_rgb, axis=0) if test_rgb else train_patches

    def _features(patches: np.ndarray) -> np.ndarray:
        flat = patches.reshape(patches.shape[0], -1, 3)
        centered, _ = remove_dc(rgb_to_opponent(flat))
        return centered.reshape(patches.shape[0], -1)

    train_feats = _features(train_patches)
    test_feats = _features(test_patches)
    whitening = fit_whitening(train_feats, input_manifest_sha256=None)  # train ONLY
    report = build_audit_report(
        rgb_patches=train_patches,
        train_features=train_feats,
        test_features=test_feats,
        whitening=whitening,
        dataset="synthetic",
    )
    try:
        write_audit_report(report, out, overwrite=force)
    except FileExistsError as exc:
        typer.echo(f"{exc}\n(use --force to overwrite)", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"wrote data audit report to {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
