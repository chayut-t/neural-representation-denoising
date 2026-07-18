"""Canary tests for the public leak scanner (plan §2.4, Phase 2 task 10).

We test the scanner's regexes against SYNTHETIC canaries — never real private
identifiers — and confirm it passes on innocuous text.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCANNER = Path(__file__).resolve().parents[2] / "scripts" / "release" / "scan_public_leaks.py"
_spec = importlib.util.spec_from_file_location("scan_public_leaks", _SCANNER)
assert _spec and _spec.loader
scanner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scanner)


# Synthetic canaries (fake values) that MUST be flagged.
CANARIES = [
    "image: 000000000000.dkr.ecr.us-west-1.amazonaws.com/example:tag",
    "index = fakedomain-000000000000.d.codeartifact.us-west-1.amazonaws.com/pypi/x/simple/",
    "see https://wiki.example.internal/page",
    "path = /Users/someuser/project/file",
    "scratch = /scratch/someuser/run",
    "AWS_ACCESS_KEY_ID=AKIAAAAAAAAAAAAAAAAA",
    "Authorization: abcdefghijklmnopqrstuvwxyz123456",
]

INNOCUOUS = [
    "This plan mentions cluster, queue, and pod as ordinary words.",
    "GPU model class NVIDIA L4 with 24 GB VRAM.",
    "uv sync --locked --extra image",
    "https://pypi.org/simple/ is the public index.",
]


@pytest.mark.parametrize("line", CANARIES)
def test_canary_is_flagged(line: str) -> None:
    assert any(pat.search(line) for _, pat in scanner.PATTERNS), f"canary not caught: {line!r}"


@pytest.mark.parametrize("line", INNOCUOUS)
def test_innocuous_not_flagged(line: str) -> None:
    assert not any(pat.search(line) for _, pat in scanner.PATTERNS), f"false positive: {line!r}"


def test_scan_repo_is_clean() -> None:
    """The tracked repo must currently have zero generic-pattern hits."""
    assert scanner.scan() == []
