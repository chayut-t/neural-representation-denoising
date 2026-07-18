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
from neural_repr.provenance.leak_patterns import PATTERNS  # noqa: E402

# Extensions treated as scannable text; binaries/data are skipped.
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".cfg",
    ".ini",
    ".yaml",
    ".yml",
    ".json",
    ".csv",
    ".tex",
    ".bib",
    ".sh",
    ".cff",
    "",
}

# Paths never scanned: the git-ignored operator note is not tracked anyway, but
# guard against accidental inclusion; the pattern/canary definitions and the
# canary test (which contain SYNTHETIC identifiers by design) are allowed.
SKIP_SUBSTRINGS = (
    "infrastructure.local",
    "scripts/release/scan_public_leaks.py",
    "src/neural_repr/provenance/leak_patterns.py",
    "tests/regression/test_leak_scanner.py",
)


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return [REPO_ROOT / line for line in out.stdout.splitlines() if line]


def scan() -> list[str]:
    findings: list[str] = []
    for path in _tracked_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if any(s in rel for s in SKIP_SUBSTRINGS):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for name, pat in PATTERNS:
                if pat.search(line):
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
