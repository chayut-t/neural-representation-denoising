"""Tests for collision-failing atomic writes (plan §0.3 no-overwrite; R2 finding 5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_repr.data.io_safe import (
    OutputExistsError,
    atomic_write_bytes,
    atomic_write_text,
    write_or_verify_text,
)


def test_atomic_write_text_creates(tmp_path: Path) -> None:
    p = tmp_path / "sub" / "a.txt"
    atomic_write_text(p, "hello")
    assert p.read_text() == "hello"


def test_atomic_write_text_refuses_overwrite(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    atomic_write_text(p, "one")
    with pytest.raises(OutputExistsError, match="refusing to overwrite"):
        atomic_write_text(p, "two")
    assert p.read_text() == "one"  # unchanged
    atomic_write_text(p, "two", overwrite=True)
    assert p.read_text() == "two"


def test_atomic_write_text_leaves_no_temp_files(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    atomic_write_text(p, "x")
    # Only the target remains; no leftover .tmp files.
    assert [q.name for q in tmp_path.iterdir()] == ["a.txt"]


def test_atomic_write_bytes_refuses_overwrite(tmp_path: Path) -> None:
    p = tmp_path / "a.bin"
    atomic_write_bytes(p, b"\x00\x01")
    with pytest.raises(OutputExistsError):
        atomic_write_bytes(p, b"\x02")


def test_write_or_verify_matches_and_mismatches(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    assert write_or_verify_text(p, "same") is True  # created
    assert write_or_verify_text(p, "same") is False  # verified, no write
    with pytest.raises(OutputExistsError, match="differs"):
        write_or_verify_text(p, "different")
