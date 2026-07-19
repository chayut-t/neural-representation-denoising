#!/usr/bin/env python3
"""Public infrastructure-leak scanner (plan §2.4, Phase 2 task 10, control 2).

Scans git-tracked text files for GENERIC infrastructure/secret patterns — never
a deny-list of real private identifiers (that would itself disclose them, per the
plan). The organization-specific pattern scan is a separate PRIVATE tool run from
config outside this repo (control 3). The primary guarantee is the structured
allow-list on committed manifests/configs (control 1).

Generic patterns flagged:
- 12-digit cloud account IDs;
- private/internal registry or package-index URLs (ecr, codeartifact, .internal, .corp);
- absolute home/scratch paths (/Users/<name>, /home/<name>, /scratch/<name>);
- obvious credential assignments (AWS keys, bearer/authorization tokens).

Exit non-zero if any tracked file matches. Run: python scripts/release/scan_public_leaks.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Patterns live in the package so the container self-check shares one source of
# truth with this repo scanner (plan §2.4). Import from src/ without installing.
sys.path.insert(0, str(REPO_ROOT / "src"))
from neural_repr.provenance.leak_patterns import CANARIES, PATTERNS  # noqa: E402

# We scan every tracked file that decodes as UTF-8 text (so repository-specific
# names like Dockerfile.cpu / Dockerfile.cuda are covered), skipping only files
# whose suffix marks them as binary/data assets.
_BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".psd",
    ".ico",
    ".zip",
    ".gz",
    ".tar",
    ".whl",
    ".parquet",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
}

# EXACT tracked paths that are exempt as whole files: the scanner and its shared
# pattern module (they must contain the patterns), and the git-ignored operator
# note (guard against accidental tracking).
_SKIP_EXACT_PATHS = frozenset(
    {
        "scripts/release/scan_public_leaks.py",
        "src/neural_repr/provenance/leak_patterns.py",
        "docs/infrastructure.local.md",
    }
)


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return [REPO_ROOT / line for line in out.stdout.splitlines() if line]


def _strip_canaries(line: str) -> str:
    """Remove all known synthetic-canary substrings from a line.

    A canary-only line becomes clean; a line that ALSO contains a real identifier
    still trips the patterns after stripping. This is the narrow, line-level
    exemption the review asked for (not a whole-file skip)."""
    for canary in CANARIES:
        line = line.replace(canary, "")
    return line


def scan() -> list[str]:
    findings: list[str] = []
    for path in _tracked_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in _SKIP_EXACT_PATHS:
            continue
        if path.suffix.lower() in _BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue  # undecodable => treat as binary
        for lineno, line in enumerate(text.splitlines(), 1):
            probe = _strip_canaries(line)  # canary substrings removed; real leaks remain
            for name, pat in PATTERNS:
                if pat.search(probe):
                    findings.append(f"{rel}:{lineno}: [{name}] {line.strip()[:120]}")
    return findings


def main() -> int:
    findings = scan()
    if findings:
        print("Public leak scan FAILED — infrastructure/secret patterns found:", file=sys.stderr)
        for f in findings:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("[scan-public-leaks] no generic infrastructure/secret patterns in tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
