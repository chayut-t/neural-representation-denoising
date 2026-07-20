#!/usr/bin/env python3
"""Canonical, non-overwriting dissertation build (plan §7 Phase 2 task 9; Gate P2).

Builds the working dissertation edition to a NEW ``builds/dissertation/<build-id>/``
directory. Refuses to overwrite an existing build. Writes a build manifest recording
the source inputs (including generated artifacts and the FILE_MAP lineage) and the
frozen 2026 baseline PDF hash, so a build is traceable to exactly what produced it.

Safety properties (codex A4):
- Build IDs are validated against a narrow grammar and the resolved output directory
  is confirmed to stay under ``builds/dissertation/`` (no ``..``/absolute escapes).
- The content-addressed default ID fingerprints EVERY actual PDF input, including
  ``generated/`` figures/tables and ``FILE_MAP.csv``, so changing any of them yields a
  different default ID (no silent collision / misidentification).
- The build runs in a temporary directory and is published atomically only on
  success; a failed build leaves no ``<build-id>/`` directory.
- A failed build always returns non-zero (even with ``--keep-going``) and never
  claims a PDF was written when none exists. ``--keep-going`` only preserves the
  failed build's diagnostics under ``<build-id>.failed/`` for inspection.

Usage:
    python scripts/release/build_dissertation.py [--build-id ID] [--keep-going]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DISS = REPO_ROOT / "dissertation"
BUILDS = REPO_ROOT / "builds" / "dissertation"
BASELINE_PDF = REPO_ROOT / "legacy" / "rewrite-2026" / "source" / "thesis.pdf"

# Build IDs must be a single safe path segment: letters, digits, dot, underscore,
# hyphen — no separators, no "..", not empty. This blocks path traversal/escape.
_BUILD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sources_fingerprint() -> str:
    """Content hash over ALL dissertation inputs to the build (deterministic, sorted).

    Includes every file under ``dissertation/`` — the ``.tex`` sources, ``generated/``
    figures/tables (real PDF inputs), and ``FILE_MAP.csv`` (lineage/provenance) — so
    the content-addressed build ID changes if any actual input changes. Excludes only
    the build-output tree itself (``builds/`` is not under ``dissertation/``).
    """
    h = hashlib.sha256()
    for path in sorted(DISS.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(DISS).as_posix()
        h.update(rel.encode())
        h.update(_sha256_file(path).encode())
    return h.hexdigest()


def _validate_build_id(build_id: str) -> Path:
    """Validate the ID against the safe grammar and confirm containment under BUILDS.

    Returns the resolved output directory. Raises ValueError on an unsafe ID or a
    destination that would resolve outside ``builds/dissertation/``.
    """
    if not _BUILD_ID_RE.match(build_id):
        raise ValueError(
            f"unsafe build id {build_id!r}: must match {_BUILD_ID_RE.pattern} "
            "(one path segment; no separators, no '..', non-empty)"
        )
    out_dir = (BUILDS / build_id).resolve()
    builds_root = BUILDS.resolve()
    if out_dir.parent != builds_root:
        raise ValueError(f"resolved build dir {out_dir} escapes {builds_root}")
    return out_dir


def build(build_id: str | None, *, keep_going: bool) -> int:
    if build_id is None:
        build_id = "src-" + _sources_fingerprint()[:16]
    try:
        out_dir = _validate_build_id(build_id)
    except ValueError as exc:
        print(f"invalid build id: {exc}", file=sys.stderr)
        return 1

    if out_dir.exists():
        print(f"refusing to overwrite existing build directory: {out_dir}", file=sys.stderr)
        return 1

    BUILDS.mkdir(parents=True, exist_ok=True)
    # Build into a temp dir; publish atomically only on success.
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=BUILDS))
    try:
        cmd = ["latexmk", "-r", "latexmkrc", f"-outdir={tmp_dir}", "main.tex"]
        result = subprocess.run(cmd, cwd=DISS)
        pdf = tmp_dir / "main.pdf"
        succeeded = result.returncode == 0 and pdf.is_file()

        manifest = {
            "build_id": build_id,
            "sources_fingerprint": "sha256:" + _sources_fingerprint(),
            "file_map_sha256": "sha256:" + _sha256_file(DISS / "FILE_MAP.csv"),
            "baseline_2026_pdf_sha256": "sha256:" + _sha256_file(BASELINE_PDF),
            "pdf_sha256": ("sha256:" + _sha256_file(pdf)) if pdf.is_file() else None,
            "latexmk_exit": result.returncode,
            "succeeded": succeeded,
        }
        (tmp_dir / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        if not succeeded:
            print(
                f"latexmk failed (exit {result.returncode}); "
                f"{'PDF present' if pdf.is_file() else 'no PDF produced'}",
                file=sys.stderr,
            )
            if keep_going:
                # Never delete a prior failed attempt: pick the next free unique ID
                # (plan §0.3 — no overwrite, even for diagnostics).
                attempt = 0
                failed_dir = out_dir.with_name(f"{out_dir.name}.failed-{attempt}")
                while failed_dir.exists():
                    attempt += 1
                    failed_dir = out_dir.with_name(f"{out_dir.name}.failed-{attempt}")
                shutil.move(str(tmp_dir), str(failed_dir))
                print(
                    f"[build-dissertation] preserved diagnostics in {failed_dir}", file=sys.stderr
                )
            # Always non-zero on failure, regardless of --keep-going.
            return 1

        # Success: publish atomically.
        shutil.move(str(tmp_dir), str(out_dir))
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[build-dissertation] wrote {out_dir.relative_to(REPO_ROOT)}/main.pdf")
    print(f"[build-dissertation] build-id: {build_id}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Non-overwriting dissertation build.")
    parser.add_argument(
        "--build-id", default=None, help="Explicit build ID (default: content-addressed)."
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Preserve failed-build diagnostics under <build-id>.failed/ (still exits non-zero).",
    )
    args = parser.parse_args()
    return build(args.build_id, keep_going=args.keep_going)


if __name__ == "__main__":
    raise SystemExit(main())
