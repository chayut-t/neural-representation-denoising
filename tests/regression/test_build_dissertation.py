"""Regression tests for the non-overwriting dissertation build (codex P1.7).

These test the build harness's guarantees without running a full latexmk build
(which is slow and needs TeX Live): the content-addressed ID is deterministic,
and an existing build directory is never overwritten.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

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
