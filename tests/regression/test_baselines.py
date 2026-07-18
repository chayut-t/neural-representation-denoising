"""Regression test: the immutable legacy baselines match their manifests (plan §2)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_VERIFY = Path(__file__).resolve().parents[2] / "scripts" / "release" / "verify_baselines.py"
_spec = importlib.util.spec_from_file_location("verify_baselines", _VERIFY)
assert _spec and _spec.loader
verify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify)


def test_all_baselines_match_manifests() -> None:
    problems: list[str] = []
    for tree_rel, manifest_rel in verify.BASELINES:
        root = verify.REPO_ROOT / tree_rel
        manifest = root / manifest_rel
        assert manifest.is_file(), f"missing manifest for {tree_rel}"
        problems.extend(verify.verify_tree(root, manifest))
    assert problems == [], f"baseline drift detected: {problems}"
