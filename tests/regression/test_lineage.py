"""Regression tests for lineage completeness (plan §7 Phase 2 task 9; codex P1.5).

Each test reshapes an in-memory copy of the repo (dissertation tree + 2026 source +
FILE_MAP) in a tmp dir, points the checker's module-level paths at it, and asserts the
checker FAILS for that specific corruption.
"""

from __future__ import annotations

import csv
import importlib.util
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_MOD = REPO_ROOT / "scripts" / "release" / "check_lineage.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_lineage", _MOD)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clone_repo(tmp: Path) -> None:
    """Copy the minimal trees the checker reads into tmp."""
    (tmp / "docs").mkdir(parents=True)
    (tmp / "legacy" / "rewrite-2026").mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "dissertation", tmp / "dissertation")
    shutil.copytree(
        REPO_ROOT / "legacy" / "rewrite-2026" / "source", tmp / "legacy" / "rewrite-2026" / "source"
    )
    shutil.copy2(
        REPO_ROOT / "legacy" / "rewrite-2026" / "MANIFEST.sha256",
        tmp / "legacy" / "rewrite-2026" / "MANIFEST.sha256",
    )
    shutil.copy2(
        REPO_ROOT / "docs" / "rewrite-2026-inventory.csv",
        tmp / "docs" / "rewrite-2026-inventory.csv",
    )


def _point_checker_at(mod, tmp: Path) -> None:
    mod.REPO_ROOT = tmp
    mod.SRC_2026 = tmp / "legacy" / "rewrite-2026" / "source"
    mod.MANIFEST_2026 = tmp / "legacy" / "rewrite-2026" / "MANIFEST.sha256"
    mod.INVENTORY = tmp / "docs" / "rewrite-2026-inventory.csv"
    mod.DISS = tmp / "dissertation"
    mod.FILE_MAP = tmp / "dissertation" / "FILE_MAP.csv"
    mod.MIGRATION_EXPECTED = tmp / "dissertation" / "MIGRATION_EXPECTED.csv"


def _read_map(tmp: Path) -> tuple[list[str], list[dict]]:
    with (tmp / "dissertation" / "FILE_MAP.csv").open() as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def _write_map(tmp: Path, fields: list[str], rows: list[dict]) -> None:
    with (tmp / "dissertation" / "FILE_MAP.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def repo(tmp_path: Path):
    _clone_repo(tmp_path)
    mod = _load_checker()
    _point_checker_at(mod, tmp_path)
    # Sanity: the pristine clone must pass.
    assert mod.check_file_map() == []
    return tmp_path, mod


def test_clean_clone_passes(repo) -> None:
    _tmp, mod = repo
    assert mod.check_file_map() == []
    assert mod.check_inventory_covers_labels() == []
    assert mod.check_migration_fanout() == []


def test_delete_target_and_row_together_detected(repo) -> None:
    """R2-7: deleting a derived file AND its FILE_MAP row is caught via the frozen tuples.

    check_file_map() alone cannot see this (both the file and its row are gone), but the
    immutable MIGRATION_EXPECTED tuple for that target is still required.
    """
    tmp, mod = repo
    (tmp / "dissertation" / "chapters" / "chap1.tex").unlink()
    fields, rows = _read_map(tmp)
    rows = [r for r in rows if r["dissertation_file"] != "chapters/chap1.tex"]
    _write_map(tmp, fields, rows)

    assert mod.check_file_map() == []  # tree-derived check is blind
    problems = mod.check_migration_fanout()
    assert any("chap1.tex" in p for p in problems)


def test_delete_one_fanout_target_and_row_detected(repo) -> None:
    """R2-7: the actual one-to-many case — thesis.tex -> main + 3 preamble files.

    Deleting preamble/macros.tex AND its FILE_MAP row must be caught, even though
    thesis.tex is still cited by the other three rows (the old source-basename check
    missed this).
    """
    tmp, mod = repo
    (tmp / "dissertation" / "preamble" / "macros.tex").unlink()
    fields, rows = _read_map(tmp)
    rows = [r for r in rows if r["dissertation_file"] != "preamble/macros.tex"]
    _write_map(tmp, fields, rows)

    problems = mod.check_migration_fanout()
    assert any("preamble/macros.tex" in p for p in problems)


def test_unexpected_fanout_tuple_detected(repo) -> None:
    """An extra FILE_MAP row not in the expected tuples is flagged."""
    tmp, mod = repo
    fields, rows = _read_map(tmp)
    rows.append(
        {
            "dissertation_file": "preamble/extra.tex",
            "derivation": "translated-from",
            "source_2026_file": "thesis.tex",
            "source_sha256": rows[-1]["source_sha256"],
            "current_sha256": "0" * 64,
            "notes": "bogus",
        }
    )
    _write_map(tmp, fields, rows)
    assert any("unexpected migration tuple" in p for p in mod.check_migration_fanout())


def test_removed_row_detected(repo) -> None:
    tmp, mod = repo
    fields, rows = _read_map(tmp)
    rows = [r for r in rows if r["dissertation_file"] != "chapters/chap1.tex"]
    _write_map(tmp, fields, rows)
    assert any("chap1.tex" in p for p in mod.check_file_map())


def test_duplicate_row_detected(repo) -> None:
    tmp, mod = repo
    fields, rows = _read_map(tmp)
    rows.append(dict(rows[0]))
    _write_map(tmp, fields, rows)
    assert any("duplicate" in p for p in mod.check_file_map())


def test_unknown_derivation_detected(repo) -> None:
    tmp, mod = repo
    fields, rows = _read_map(tmp)
    rows[0]["derivation"] = "conjured-from-thin-air"
    _write_map(tmp, fields, rows)
    assert any("unknown derivation" in p for p in mod.check_file_map())


def test_bad_source_hash_detected(repo) -> None:
    tmp, mod = repo
    fields, rows = _read_map(tmp)
    rows[0]["source_sha256"] = "0" * 64
    _write_map(tmp, fields, rows)
    assert any("source hash drift" in p for p in mod.check_file_map())


def test_missing_source_detected(repo) -> None:
    tmp, mod = repo
    fields, rows = _read_map(tmp)
    rows[0]["source_2026_file"] = "does_not_exist.tex"
    _write_map(tmp, fields, rows)
    assert any("missing 2026 source" in p for p in mod.check_file_map())


def test_stale_current_hash_detected(repo) -> None:
    tmp, mod = repo
    # Edit a byte-identical copy on disk without updating FILE_MAP.
    target = tmp / "dissertation" / "chapters" / "chap1.tex"
    target.write_text(target.read_text() + "\n% drifted\n")
    assert any("current_sha256 stale" in p for p in mod.check_file_map())


def test_unexpected_target_detected(repo) -> None:
    tmp, mod = repo
    fields, rows = _read_map(tmp)
    rows.append(
        {
            "dissertation_file": "chapters/chap99.tex",
            "derivation": "translated-from",
            "source_2026_file": "thesis.tex",
            "source_sha256": rows[-1]["source_sha256"],
            "current_sha256": "0" * 64,
            "notes": "bogus",
        }
    )
    _write_map(tmp, fields, rows)
    assert any("unexpected/undeclared target" in p for p in mod.check_file_map())
