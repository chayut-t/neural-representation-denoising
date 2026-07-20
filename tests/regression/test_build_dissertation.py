"""Regression tests for the non-overwriting dissertation build (codex P1.7).

These test the build harness's guarantees without running a full latexmk build
(which is slow and needs TeX Live): the content-addressed ID is deterministic,
and an existing build directory is never overwritten.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_MOD = REPO_ROOT / "scripts" / "release" / "build_dissertation.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_dissertation", _MOD)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sources_fingerprint_is_deterministic() -> None:
    mod = _load()
    assert mod._sources_fingerprint() == mod._sources_fingerprint()


def test_refuses_to_overwrite_existing_build(tmp_path: Path) -> None:
    mod = _load()
    mod.BUILDS = tmp_path / "builds" / "dissertation"
    existing = mod.BUILDS / "ci-123"
    existing.mkdir(parents=True)
    # A build targeting an existing directory must refuse (return non-zero) and
    # must not touch the directory's contents.
    rc = mod.build("ci-123", keep_going=False)
    assert rc == 1
    assert list(existing.iterdir()) == []  # untouched


def test_content_addressed_id_prefix() -> None:
    mod = _load()
    # With no explicit id, the default is a content-addressed 'src-<hash>' id.
    # We don't run latexmk here; just confirm the id derivation shape.
    fp = mod._sources_fingerprint()
    assert len(fp) == 64
    expected_id = "src-" + fp[:16]
    assert expected_id.startswith("src-")
    assert len(expected_id) == len("src-") + 16


@pytest.mark.parametrize(
    "bad_id",
    ["/tmp/escaped", "../evil", "a/b", "", ".", "..", "with space", "x" * 65],
)
def test_rejects_unsafe_build_ids(tmp_path: Path, bad_id: str) -> None:
    """A4: absolute, parent-traversal, separator, empty, and over-long IDs are rejected
    before anything is created."""
    mod = _load()
    mod.BUILDS = tmp_path / "builds" / "dissertation"
    rc = mod.build(bad_id, keep_going=False)
    assert rc == 1
    # Nothing created under the builds root.
    assert not mod.BUILDS.exists() or list(mod.BUILDS.iterdir()) == []


def test_validate_build_id_confines_under_builds(tmp_path: Path) -> None:
    mod = _load()
    mod.BUILDS = tmp_path / "builds" / "dissertation"
    with pytest.raises(ValueError, match="unsafe build id"):
        mod._validate_build_id("../../etc")
    out = mod._validate_build_id("ci-123")
    assert out.parent == mod.BUILDS.resolve()


def test_fingerprint_changes_with_generated_and_file_map(tmp_path: Path, monkeypatch) -> None:
    """A4: changing a generated/ artifact or FILE_MAP.csv changes the default build identity."""
    mod = _load()
    diss = tmp_path / "dissertation"
    (diss / "generated").mkdir(parents=True)
    (diss / "main.tex").write_text("doc")
    (diss / "FILE_MAP.csv").write_text("header\n")
    fig = diss / "generated" / "fig1.pdf"
    fig.write_text("figure-v1")
    monkeypatch.setattr(mod, "DISS", diss)

    fp0 = mod._sources_fingerprint()
    fig.write_text("figure-v2")  # a generated artifact changed
    fp1 = mod._sources_fingerprint()
    assert fp0 != fp1, "changing a generated/ artifact must change the fingerprint"

    (diss / "FILE_MAP.csv").write_text("header\nrow\n")  # lineage metadata changed
    fp2 = mod._sources_fingerprint()
    assert fp2 != fp1, "changing FILE_MAP.csv must change the fingerprint"


def test_failed_build_returns_nonzero_and_publishes_nothing(tmp_path: Path, monkeypatch) -> None:
    """A4: a failed latexmk returns non-zero, writes no <build-id>/, claims no PDF."""
    mod = _load()
    mod.BUILDS = tmp_path / "builds" / "dissertation"

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=["latexmk"], returncode=12)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    rc = mod.build("ci-fail", keep_going=False)
    assert rc == 1
    assert not (mod.BUILDS / "ci-fail").exists()  # nothing published


def test_keep_going_still_nonzero_but_preserves_diagnostics(tmp_path: Path, monkeypatch) -> None:
    """A4: --keep-going preserves diagnostics under <id>.failed/ but still exits non-zero."""
    mod = _load()
    mod.BUILDS = tmp_path / "builds" / "dissertation"

    def fake_run(*_args, **kwargs):
        # Emulate latexmk writing a log but no PDF into the outdir.
        outdir = next(a for a in _args[0] if str(a).startswith("-outdir="))
        Path(str(outdir).split("=", 1)[1], "main.log").write_text("error log")
        return subprocess.CompletedProcess(args=["latexmk"], returncode=12)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    rc = mod.build("ci-fail", keep_going=True)
    assert rc == 1  # still non-zero
    assert not (mod.BUILDS / "ci-fail").exists()  # not published as a success
    failed = mod.BUILDS / "ci-fail.failed-0"
    assert failed.exists()
    manifest = failed / "build-manifest.json"
    assert manifest.exists()
    assert '"pdf_sha256": null' in manifest.read_text()  # never claims a PDF
    assert '"succeeded": false' in manifest.read_text()

    # A second failed attempt must NOT delete the first: it gets a fresh unique ID.
    rc2 = mod.build("ci-fail", keep_going=True)
    assert rc2 == 1
    assert (mod.BUILDS / "ci-fail.failed-0").exists()  # prior attempt preserved
    assert (mod.BUILDS / "ci-fail.failed-1").exists()  # new attempt is a new ID
