#!/usr/bin/env python3
"""Verify the immutable legacy baselines against their committed manifests.

Enforces the hard preservation rule (plan §2, §0.3): the 2016 dissertation and the
2026 rewrite trees must remain byte-identical to their ``MANIFEST.sha256`` files.
Exits non-zero on any mismatch, missing file, or unlisted extra file.

Usage: python scripts/release/verify_baselines.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# (tree root relative to repo, manifest path relative to that root)
BASELINES = [
    ("legacy/dissertation-2016", "MANIFEST.sha256"),
    ("legacy/rewrite-2026", "MANIFEST.sha256"),
]

# Provenance meta-docs that a manifest may legitimately not list (the 2016
# manifest, frozen in Phase 0, scopes itself to historical materials and omits
# its own README). Any OTHER unlisted file is treated as an unrecorded mutation.
ALLOWED_UNLISTED = {"README.md"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_tree(root: Path, manifest: Path) -> list[str]:
    """Return a list of problems (empty == clean)."""
    problems: list[str] = []
    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        expected, _, rel = line.partition("  ")
        rel = rel.strip()
        if not rel:
            problems.append(f"malformed manifest line: {line!r}")
            continue
        listed.add(rel)
        target = root / rel
        if not target.is_file():
            problems.append(f"missing file: {root.name}/{rel}")
            continue
        actual = _sha256(target)
        if actual != expected:
            problems.append(f"hash mismatch: {root.name}/{rel}")
    # Detect files present in the tree but absent from the manifest (excluding the
    # manifest itself), which would mean an unrecorded mutation.
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == manifest.name:
            continue
        rel = path.relative_to(root).as_posix()
        if rel not in listed and rel not in ALLOWED_UNLISTED:
            problems.append(f"unlisted extra file: {root.name}/{rel}")
    return problems


def main() -> int:
    all_problems: list[str] = []
    for tree_rel, manifest_rel in BASELINES:
        root = REPO_ROOT / tree_rel
        manifest = root / manifest_rel
        if not manifest.is_file():
            all_problems.append(f"missing manifest: {tree_rel}/{manifest_rel}")
            continue
        problems = verify_tree(root, manifest)
        status = "OK" if not problems else f"{len(problems)} problem(s)"
        print(f"[verify-baselines] {tree_rel}: {status}")
        all_problems.extend(problems)

    if all_problems:
        print("\nBaseline verification FAILED:", file=sys.stderr)
        for p in all_problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("[verify-baselines] all baselines match their manifests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
