"""Collision-failing, atomic file writes (plan §0.3 no-overwrite rule).

Pipeline writers must not silently overwrite an existing artifact: results use new
IDs or content-addressed/versioned paths, and an unexpected collision is a failure,
not a clobber. :func:`atomic_write_text` writes to a temp file in the same directory
and atomically renames it into place, refusing to replace an existing target unless
the caller explicitly opts in (e.g. deterministic-regeneration verification, which
should compare rather than overwrite — see :func:`write_or_verify_text`).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


class OutputExistsError(FileExistsError):
    """Raised when an output path already exists and overwrite was not permitted."""


def atomic_write_text(path: Path, text: str, *, overwrite: bool = False) -> None:
    """Atomically write ``text`` to ``path``; refuse to clobber unless ``overwrite``.

    Writes to a temp file in the destination directory then ``os.replace`` (atomic on
    the same filesystem). With ``overwrite=False`` (default) an existing target raises
    :class:`OutputExistsError` before anything is written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise OutputExistsError(
            f"refusing to overwrite existing output: {path} "
            "(use a new/versioned path, or pass overwrite=True for deliberate replacement)"
        )
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_bytes(path: Path, data: bytes, *, overwrite: bool = False) -> None:
    """Atomically write ``data`` to ``path``; refuse to clobber unless ``overwrite``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise OutputExistsError(
            f"refusing to overwrite existing output: {path} "
            "(use a new/versioned path, or pass overwrite=True for deliberate replacement)"
        )
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_or_verify_text(path: Path, text: str) -> bool:
    """Write ``text`` if ``path`` is absent; if present, VERIFY it matches (no overwrite).

    Returns True if it wrote a new file, False if an existing file already matched.
    Raises :class:`OutputExistsError` if an existing file differs — this is how
    deterministic regeneration is checked without clobbering the committed artifact.
    """
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return False
        raise OutputExistsError(
            f"existing output {path} differs from freshly generated content "
            "(deterministic regeneration mismatch)"
        )
    atomic_write_text(path, text, overwrite=False)
    return True
