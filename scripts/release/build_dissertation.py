#!/usr/bin/env python3
"""Canonical, non-overwriting dissertation build (plan §7 Phase 2 task 9; Gate P2).

Builds the working dissertation edition to a NEW ``builds/dissertation/<build-id>/``
directory. Refuses to overwrite an existing build. Writes a build manifest recording
the source inputs (FILE_MAP hash) and the frozen 2026 baseline PDF hash the plan requires,
so a build is traceable to exactly what produced it.

Usage:
    python scripts/release/build_dissertation.py [--build-id ID] [--keep-going]

If ``--build-id`` is omitted, a content-addressed ID is derived from the current
dissertation sources (so identical sources reuse the same ID and a second run with
unchanged sources fails as a collision — proving non-overwriting).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DISS = REPO_ROOT / "dissertation"
BUILDS = REPO_ROOT / "builds" / "dissertation"
BASELINE_PDF = REPO_ROOT / "legacy" / "rewrite-2026" / "source" / "thesis.pdf"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sources_fingerprint() -> str:
    """Content hash over all dissertation source inputs (deterministic, sorted)."""
    h = hashlib.sha256()
    for path in sorted(DISS.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(DISS).as_posix()
        if rel.startswith("generated/") or rel == "FILE_MAP.csv":
            # generated/ is produced content; FILE_MAP is metadata, hashed separately.
            continue
        h.update(rel.encode())
        h.update(_sha256_file(path).encode())
    return h.hexdigest()


def build(build_id: str | None, *, keep_going: bool) -> int:
    if build_id is None:
        build_id = "src-" + _sources_fingerprint()[:16]

    out_dir = BUILDS / build_id
    if out_dir.exists():
        print(f"refusing to overwrite existing build directory: {out_dir}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True)

    # Build with latexmk -> LuaLaTeX + biber, writing all products into out_dir.
    cmd = [
        "latexmk",
        "-r",
        "latexmkrc",
        f"-outdir={out_dir}",
        "main.tex",
    ]
    result = subprocess.run(cmd, cwd=DISS)
    pdf = out_dir / "main.pdf"
    if result.returncode != 0 or not pdf.is_file():
        print(f"latexmk failed (exit {result.returncode}); no PDF produced", file=sys.stderr)
        if not keep_going:
            return 1

    # Build manifest: source inputs + frozen baseline PDF hash (plan §6.1 / task 9).
    manifest = {
        "build_id": build_id,
        "sources_fingerprint": "sha256:" + _sources_fingerprint(),
        "file_map_sha256": "sha256:" + _sha256_file(DISS / "FILE_MAP.csv"),
        "baseline_2026_pdf_sha256": "sha256:" + _sha256_file(BASELINE_PDF),
        "pdf_sha256": "sha256:" + _sha256_file(pdf) if pdf.is_file() else None,
        "latexmk_exit": result.returncode,
    }
    (out_dir / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[build-dissertation] wrote {out_dir.relative_to(REPO_ROOT)}/main.pdf")
    print(f"[build-dissertation] build-id: {build_id}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Non-overwriting dissertation build.")
    parser.add_argument(
        "--build-id", default=None, help="Explicit build ID (default: content-addressed)."
    )
    parser.add_argument(
        "--keep-going", action="store_true", help="Write the manifest even if latexmk fails."
    )
    args = parser.parse_args()
    return build(args.build_id, keep_going=args.keep_going)


if __name__ == "__main__":
    raise SystemExit(main())
