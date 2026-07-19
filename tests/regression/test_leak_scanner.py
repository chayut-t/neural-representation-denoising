"""Canary tests for the public leak scanner (plan §2.4, Phase 2 task 10; codex P1.6).

Canary values come from the package (single source of truth) — never hand-written
private identifiers. We confirm: every canary is caught by the patterns; innocuous
text is not; the scanner covers repository-specific text names (Dockerfiles); the
line-level canary exemption still catches a real identifier on a canary-bearing line;
and the tracked repo is clean.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from neural_repr.provenance.leak_patterns import CANARIES, canaries_all_detected

_SCANNER = Path(__file__).resolve().parents[2] / "scripts" / "release" / "scan_public_leaks.py"
_spec = importlib.util.spec_from_file_location("scan_public_leaks", _SCANNER)
assert _spec and _spec.loader
scanner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scanner)


INNOCUOUS = [
    "This plan mentions cluster, queue, and pod as ordinary words.",
    "GPU model class NVIDIA L4 with 24 GB VRAM.",
    "uv sync --locked --extra image",
    "https://pypi.org/simple/ is the public index.",
]


@pytest.mark.parametrize("canary", CANARIES)
def test_canary_is_flagged(canary: str) -> None:
    assert any(pat.search(canary) for _, pat in scanner.PATTERNS), f"canary not caught: {canary!r}"


@pytest.mark.parametrize("line", INNOCUOUS)
def test_innocuous_not_flagged(line: str) -> None:
    assert not any(pat.search(line) for _, pat in scanner.PATTERNS), f"false positive: {line!r}"


def test_package_canaries_all_detected() -> None:
    """The package-level canary self-check (used by check-image) must pass."""
    assert canaries_all_detected()


def test_strip_canaries_leaves_real_identifier() -> None:
    """A real (non-canary) identifier on a canary-bearing line must still be caught.

    The synthetic 'real' identifier is assembled at runtime from fragments so no
    literal ECR-shaped string sits in this tracked file (which the scanner reads)."""
    real = "9" * 12 + ".dkr.ecr." + "eu-west-9" + ".amazonaws.com/x:y"
    line = f"{CANARIES[0]} and also {real}"
    probe = scanner._strip_canaries(line)
    assert any(pat.search(probe) for _, pat in scanner.PATTERNS)
    # Sanity: with only the canary present, the stripped line is clean.
    assert not any(pat.search(scanner._strip_canaries(CANARIES[0])) for _, pat in scanner.PATTERNS)


def test_scanner_covers_dockerfiles() -> None:
    """Dockerfile.* must be in scope (regression for the old suffix allow-list gap)."""
    scanned = {
        p.relative_to(scanner.REPO_ROOT).as_posix()
        for p in scanner._tracked_files()
        if p.suffix.lower() not in scanner._BINARY_SUFFIXES
    }
    assert "containers/Dockerfile.cpu" in scanned
    assert "containers/Dockerfile.cuda" in scanned


def test_scan_repo_is_clean() -> None:
    """The tracked repo must currently have zero generic-pattern hits."""
    assert scanner.scan() == []
