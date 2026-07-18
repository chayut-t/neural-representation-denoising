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

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# (name, compiled regex). Deliberately generic; extend via review, not with real values.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("cloud-account-id", re.compile(r"\b\d{12}\b\.dkr\.ecr\.")),
    ("ecr-registry", re.compile(r"\d+\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com")),
    ("codeartifact-url", re.compile(r"[a-z0-9-]+\.d\.codeartifact\.[a-z0-9-]+\.amazonaws\.com")),
    ("internal-domain", re.compile(r"https?://[a-z0-9.-]+\.(?:internal|corp|a2z\.com)\b")),
    ("abs-home-path", re.compile(r"/(?:Users|home)/[a-z][a-z0-9_-]+/", re.IGNORECASE)),
    ("abs-scratch-path", re.compile(r"/scratch/[a-z][a-z0-9_-]+")),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("bearer-token", re.compile(r"(?i)\b(?:authorization|bearer)\b\s*[:=]\s*[A-Za-z0-9._-]{20,}")),
]

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
# guard against accidental inclusion; the scanner's own pattern strings are allowed.
SKIP_SUBSTRINGS = ("infrastructure.local", "scripts/release/scan_public_leaks.py")


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
