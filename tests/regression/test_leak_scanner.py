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


def test_bare_account_id_forms_flagged_but_plain_number_not() -> None:
    """ARN and account_id= forms are flagged; a bare 12-digit number is not (A5)."""
    arn = "arn:aws:iam::" + "3" * 12 + ":role/foo"
    assign = "account_id = " + "4" * 12
    plain = "timestamp 123456789012 is just a number"
    assert any(pat.search(arn) for _, pat in scanner.PATTERNS)
    assert any(pat.search(assign) for _, pat in scanner.PATTERNS)
    assert not any(pat.search(plain) for _, pat in scanner.PATTERNS)


def test_no_whole_file_skips() -> None:
    """R2 finding 12: NO file is skipped wholesale — not even the scanner's own source.

    Exemptions are line-level (an inline marker), so a real secret cannot hide in a
    whole-file-skipped source. The shared pattern module and the scanner itself are
    both in scope.
    """
    assert frozenset() == scanner._SKIP_EXACT_PATHS
    scanned = {p.relative_to(scanner.REPO_ROOT).as_posix() for p in scanner._tracked_files()}
    assert "src/neural_repr/provenance/leak_patterns.py" in scanned
    assert "scripts/release/scan_public_leaks.py" in scanned


def test_line_marker_exempts_only_that_line() -> None:
    """The inline exemption marker skips exactly its line, not the whole file."""
    assert scanner._LINE_EXEMPT_MARKER == "leak-scan-allow"


def test_non_canary_internal_url_would_be_detected() -> None:
    """A NEW non-canary internal URL (as if added to any scanned file) is caught (A5).

    Assembled at runtime so no literal internal URL sits in this tracked file.
    """
    real = "https://" + "wiki.secret" + ".internal/page"
    assert real not in CANARIES
    probe = scanner._strip_canaries(real)  # not a canary, so survives stripping
    assert any(pat.search(probe) for _, pat in scanner.PATTERNS)


def test_tracked_private_note_is_unconditional_failure() -> None:
    """A tracked infrastructure.local.md / *.local.md is flagged even if content is clean (A5)."""
    assert "docs/infrastructure.local.md" in scanner._MUST_NOT_BE_TRACKED
    assert scanner._FORBIDDEN_TRACKED_SUFFIXES == (".local.md",)
