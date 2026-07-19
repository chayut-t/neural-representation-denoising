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
    assert mod.check_manifest_sources_are_mapped() == []


def test_delete_target_and_row_together_detected(repo) -> None:
    """A2: deleting a derived file AND its FILE_MAP row is caught via the frozen manifest.

    check_file_map() alone cannot see this (both the file and its row are gone, so the
    tree-derived expected set no longer lists it), but the manifest still cites chap1.tex.
    """
    tmp, mod = repo
    # Remove the derived target file...
    (tmp / "dissertation" / "chapters" / "chap1.tex").unlink()
    # ...and its FILE_MAP row.
    fields, rows = _read_map(tmp)
    rows = [r for r in rows if r["dissertation_file"] != "chapters/chap1.tex"]
    _write_map(tmp, fields, rows)

    # The tree-derived check is now blind to the loss...
    assert mod.check_file_map() == []
    # ...but the manifest-anchored check catches it.
    problems = mod.check_manifest_sources_are_mapped()
    assert any("chap1.tex" in p for p in problems)


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
