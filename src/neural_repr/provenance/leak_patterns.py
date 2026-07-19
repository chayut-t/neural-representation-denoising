"""Generic infrastructure/secret leak patterns (plan §2.4, Phase 2 task 10).

Shared by the repo scanner (``scripts/release/scan_public_leaks.py``, control 2)
and the container self-check (``neural-repr-verify check-image``). Deliberately
GENERIC — never a deny-list of real private identifiers, which would itself
disclose them. Extend via review.
"""

from __future__ import annotations

import re

# (name, compiled regex). Generic patterns only.
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

# Synthetic canaries (FAKE values) the patterns must catch — used by the
# container self-check to prove the scanner is wired and working, and by the repo
# scanner to exempt these exact synthetic substrings line-by-line (so a canary line
# is clean but a real identifier on the same line still trips).
CANARIES: tuple[str, ...] = (
    "000000000000.dkr.ecr.us-west-1.amazonaws.com/example:tag",
    "fakedomain-000000000000.d.codeartifact.us-west-1.amazonaws.com/pypi/x/simple/",
    "https://wiki.example.internal/page",
    "/Users/someuser/project/file",
    "/scratch/someuser/run",
    "AKIAAAAAAAAAAAAAAAAA",
    "Authorization: abcdefghijklmnopqrstuvwxyz123456",
)


def line_has_leak(line: str) -> str | None:
    """Return the name of the first matching pattern, or None."""
    for name, pat in PATTERNS:
        if pat.search(line):
            return name
    return None


def canaries_all_detected() -> bool:
    """True iff every synthetic canary is caught by at least one pattern."""
    return all(line_has_leak(c) is not None for c in CANARIES)
