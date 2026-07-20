"""Tests for the neural-repr-data CLI (Phase 4 task 2)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from neural_repr.cli import data as data_cli

runner = CliRunner()


def test_list_shows_registered_datasets() -> None:
    result = runner.invoke(data_cli.app, ["list"])
    assert result.exit_code == 0
    assert "div2k" in result.output
    assert "bsds500" in result.output
    assert "--accept-academic-research-terms" in result.output


def _terms_file(tmp_path: Path) -> Path:
    tf = tmp_path / "terms.txt"
    tf.write_text("DIV2K academic research terms (captured copy).")
    return tf


def test_accept_terms_rejects_wrong_token(tmp_path: Path) -> None:
    result = runner.invoke(
        data_cli.app,
        [
            "accept-terms",
            "div2k",
            "--token",
            "--accept-noncommercial-research-terms",  # wrong token for div2k
            "--terms-file",
            str(_terms_file(tmp_path)),
            "--at",
            "2026-07-19T00:00:00Z",
            "--records-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert not list(tmp_path.glob("*.terms-acceptance.json"))


def test_accept_terms_rejects_bad_timestamp(tmp_path: Path) -> None:
    result = runner.invoke(
        data_cli.app,
        [
            "accept-terms",
            "div2k",
            "--token",
            "--accept-academic-research-terms",
            "--terms-file",
            str(_terms_file(tmp_path)),
            "--at",
            "yesterday",
            "--records-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1


def test_accept_terms_writes_local_record_with_terms_hash(tmp_path: Path) -> None:
    result = runner.invoke(
        data_cli.app,
        [
            "accept-terms",
            "div2k",
            "--token",
            "--accept-academic-research-terms",
            "--terms-file",
            str(_terms_file(tmp_path)),
            "--at",
            "2026-07-19T00:00:00Z",
            "--records-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    rec = tmp_path / "div2k.terms-acceptance.json"
    assert rec.is_file()
    import json

    assert json.loads(rec.read_text())["terms_sha256"].startswith("sha256:")


def test_make_fixture_and_check(tmp_path: Path) -> None:
    out_dir = tmp_path / "synth"
    manifest = tmp_path / "synthetic.csv"
    result = runner.invoke(
        data_cli.app,
        [
            "make-fixture",
            "--out-dir",
            str(out_dir),
            "--manifest",
            str(manifest),
            "--n",
            "6",
            "--size",
            "24",
            "--seed",
            "0",
        ],
    )
    assert result.exit_code == 0
    assert manifest.is_file()
    assert len(list(out_dir.glob("*.png"))) == 6

    check = runner.invoke(
        data_cli.app, ["check", "synthetic", "--manifest", str(manifest), "--root", str(out_dir)]
    )
    assert check.exit_code == 0
    assert "valid" in check.output


def test_make_fixture_refuses_overwrite_without_force(tmp_path: Path) -> None:
    out_dir = tmp_path / "synth"
    manifest = tmp_path / "synthetic.csv"
    args = [
        "make-fixture",
        "--out-dir",
        str(out_dir),
        "--manifest",
        str(manifest),
        "--n",
        "3",
        "--size",
        "24",
    ]
    assert runner.invoke(data_cli.app, args).exit_code == 0
    # Second run without --force must refuse (no clobber).
    again = runner.invoke(data_cli.app, args)
    assert again.exit_code == 1
    # With --force it regenerates.
    assert runner.invoke(data_cli.app, [*args, "--force"]).exit_code == 0


def test_make_fixture_deterministic(tmp_path: Path) -> None:
    from neural_repr.data.records import sha256_file

    def build(sub: str) -> str:
        out_dir = tmp_path / sub
        manifest = tmp_path / f"{sub}.csv"
        runner.invoke(
            data_cli.app,
            [
                "make-fixture",
                "--out-dir",
                str(out_dir),
                "--manifest",
                str(manifest),
                "--n",
                "4",
                "--size",
                "24",
                "--seed",
                "0",
            ],
        )
        return sha256_file(out_dir / "synth_0000.png")

    assert build("a") == build("b")  # byte-identical fixture across runs


def test_audit_writes_report(tmp_path: Path) -> None:
    import json

    out = tmp_path / "audit.json"
    result = runner.invoke(data_cli.app, ["audit", "--out", str(out), "--n", "6", "--size", "40"])
    assert result.exit_code == 0
    assert out.is_file()
    report = json.loads(out.read_text())
    assert report["dataset"] == "synthetic"
    assert "whitening_covariance" in report
    assert "train_test_similarity" in report


def test_check_missing_manifest_fails(tmp_path: Path) -> None:
    result = runner.invoke(
        data_cli.app, ["check", "synthetic", "--manifest", str(tmp_path / "nope.csv")]
    )
    assert result.exit_code == 1


def test_check_unknown_dataset_fails(tmp_path: Path) -> None:
    result = runner.invoke(data_cli.app, ["check", "imagenet"])
    assert result.exit_code == 1
    assert "unknown dataset" in result.output


def test_check_div2k_cannot_read_synthetic_manifest(tmp_path: Path) -> None:
    """R2 finding 1: `check div2k` must not accept the synthetic manifest as DIV2K.

    The synthetic manifest is all-role 'train' with split IDs 'synth-*', which do not
    match DIV2K's expected roles/IDs, so checking it as div2k must fail.
    """
    out_dir = tmp_path / "synth"
    manifest = tmp_path / "synthetic.csv"
    runner.invoke(
        data_cli.app,
        [
            "make-fixture",
            "--out-dir",
            str(out_dir),
            "--manifest",
            str(manifest),
            "--n",
            "4",
            "--size",
            "24",
        ],
    )
    result = runner.invoke(data_cli.app, ["check", "div2k", "--manifest", str(manifest)])
    assert result.exit_code == 1  # roles/IDs don't match DIV2K


def test_check_on_disk_detects_tampered_bytes(tmp_path: Path) -> None:
    out_dir = tmp_path / "synth"
    manifest = tmp_path / "synthetic.csv"
    runner.invoke(
        data_cli.app,
        [
            "make-fixture",
            "--out-dir",
            str(out_dir),
            "--manifest",
            str(manifest),
            "--n",
            "4",
            "--size",
            "24",
        ],
    )
    # Clean on-disk check passes.
    ok = runner.invoke(
        data_cli.app, ["check", "synthetic", "--manifest", str(manifest), "--root", str(out_dir)]
    )
    assert ok.exit_code == 0
    # Tamper with a file's bytes -> on-disk check fails.
    (out_dir / "synth_0000.png").write_bytes(b"tampered")
    bad = runner.invoke(
        data_cli.app, ["check", "synthetic", "--manifest", str(manifest), "--root", str(out_dir)]
    )
    assert bad.exit_code == 1


def test_verify_archive_rejects_unknown_archive(tmp_path: Path) -> None:
    f = tmp_path / "x.zip"
    f.write_bytes(b"x")
    result = runner.invoke(data_cli.app, ["verify-archive", "div2k", "not_official.zip", str(f)])
    assert result.exit_code == 1
    assert "not an official" in result.output


def test_verify_archive_record_then_verify(tmp_path: Path) -> None:
    records = tmp_path / "records"
    # Pinning requires a prior terms-acceptance record.
    from neural_repr.data.records import TermsAcceptance, write_terms_acceptance

    write_terms_acceptance(
        records,
        TermsAcceptance(
            "div2k",
            "https://data.vision.ee.ethz.ch/cvl/DIV2K/",
            "--accept-academic-research-terms",
            "sha256:" + "a" * 64,
            "2026-07-19T00:00:00Z",
        ),
    )
    archive = tmp_path / "DIV2K_train_HR.zip"
    archive.write_bytes(b"pretend archive")
    rec = runner.invoke(
        data_cli.app,
        [
            "verify-archive",
            "div2k",
            "DIV2K_train_HR.zip",
            str(archive),
            "--records-dir",
            str(records),
            "--record",
        ],
    )
    assert rec.exit_code == 0
    ok = runner.invoke(
        data_cli.app,
        [
            "verify-archive",
            "div2k",
            "DIV2K_train_HR.zip",
            str(archive),
            "--records-dir",
            str(records),
        ],
    )
    assert ok.exit_code == 0
    # Tamper -> verify fails.
    archive.write_bytes(b"tampered")
    bad = runner.invoke(
        data_cli.app,
        [
            "verify-archive",
            "div2k",
            "DIV2K_train_HR.zip",
            str(archive),
            "--records-dir",
            str(records),
        ],
    )
    assert bad.exit_code == 1
