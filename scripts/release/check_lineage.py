#!/usr/bin/env python3
"""Baseline-to-working-edition lineage completeness (plan §7 Phase 2 task 9/10; Gate P2).

Checks:
1. Inventory coverage — every figure/table label in the frozen 2026 source has a row in
   docs/rewrite-2026-inventory.csv (nothing dropped in the lineage).
2. Migration fan-out — the exact (source, target, derivation) tuples in the IMMUTABLE
   dissertation/MIGRATION_EXPECTED.csv must match BOTH FILE_MAP and the current tree. Because the
   expected tuples are anchored (not derived from the mutable tree/FILE_MAP), deleting one target
   of a one-to-many source (e.g. thesis.tex -> main.tex + 3 preamble files) AND its FILE_MAP row
   is still detected — the tuple for that specific target goes missing from both.
3. FILE_MAP completeness — the set of dissertation files DERIVED from the 2026 baseline is
   defined by rule; FILE_MAP must contain exactly one valid row per such file (no missing rows
   — so a deletion is detected — no duplicates, no rows for unexpected targets).
4. Per-row validity, for every derivation type:
   - derivation is a known value;
   - the 2026 source exists and its hash matches source_sha256;
   - current_sha256 matches the dissertation file's actual current hash (so unrecorded edits
     are detected);
   - for byte-identical-copy, current_sha256 == source_sha256.

Exit non-zero on any gap. Run: python scripts/release/check_lineage.py
"""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_2026 = REPO_ROOT / "legacy" / "rewrite-2026" / "source"
MANIFEST_2026 = REPO_ROOT / "legacy" / "rewrite-2026" / "MANIFEST.sha256"
INVENTORY = REPO_ROOT / "docs" / "rewrite-2026-inventory.csv"
DISS = REPO_ROOT / "dissertation"
FILE_MAP = DISS / "FILE_MAP.csv"
# Immutable expected migration fan-out: the exact (source, target, derivation) tuples
# the 2026 baseline maps to. Anchored here (not derived from the mutable tree/FILE_MAP)
# so deleting one target of a one-to-many source AND its FILE_MAP row is still detected
# (round-2 review finding 7).
MIGRATION_EXPECTED = DISS / "MIGRATION_EXPECTED.csv"

_LABEL_RE = re.compile(r"\\label\{((?:fig|tab):[^}]+)\}")
_VALID_DERIVATIONS = {"byte-identical-copy", "translated-from"}

_EXPECTED_COLUMNS = [
    "dissertation_file",
    "derivation",
    "source_2026_file",
    "source_sha256",
    "current_sha256",
    "notes",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_derived_files() -> set[str]:
    """The set of dissertation files derived from the 2026 baseline, by rule.

    Content .tex under frontmatter/chapters/appendices/preamble, the main.tex driver, and
    any .bib under bibliography/. Excludes generated/, README, latexmkrc, FILE_MAP, .gitkeep.
    """
    files: set[str] = set()
    for sub in ("frontmatter", "chapters", "appendices", "preamble"):
        for tex in (DISS / sub).glob("*.tex"):
            files.add(tex.relative_to(DISS).as_posix())
    if (DISS / "main.tex").is_file():
        files.add("main.tex")
    for bib in (DISS / "bibliography").glob("*.bib"):
        files.add(bib.relative_to(DISS).as_posix())
    return files


def _manifest_source_files() -> set[str]:
    """Basenames of ``source/*`` entries in the immutable frozen-2026 manifest."""
    names: set[str] = set()
    for line in MANIFEST_2026.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        _, _, rel = line.partition("  ")
        rel = rel.strip()
        if rel.startswith("source/") and "/images/" not in rel:
            names.add(Path(rel).name)
    return names


def _expected_tuples() -> set[tuple[str, str, str]]:
    """Immutable expected ``(source, target, derivation)`` migration tuples."""
    with MIGRATION_EXPECTED.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        expected = {
            (r["source_2026_file"], r["dissertation_file"], r["derivation"]) for r in reader
        }
    return expected


def _file_map_tuples() -> set[tuple[str, str, str]]:
    """``(source, target, derivation)`` tuples declared by FILE_MAP rows."""
    with FILE_MAP.open(encoding="utf-8") as fh:
        return {
            (r["source_2026_file"], r["dissertation_file"], r["derivation"])
            for r in csv.DictReader(fh)
        }


def check_migration_fanout() -> list[str]:
    """Enforce the exact immutable (source, target, derivation) fan-out.

    Compares the frozen expected tuples against BOTH ``FILE_MAP`` and the current tree,
    so a one-to-many source (e.g. ``thesis.tex`` -> main + 3 preamble files) is fully
    protected: deleting one of its targets AND its FILE_MAP row still fails here because
    the expected tuple for that specific target is missing from both (round-2 finding 7).
    """
    problems: list[str] = []
    expected = _expected_tuples()

    # Every required source must exist in the frozen manifest (baseline unchanged).
    manifest_names = _manifest_source_files()
    for src in sorted({s for s, _, _ in expected}):
        if src not in manifest_names:
            problems.append(f"expected source {src!r} absent from the frozen 2026 manifest")

    # FILE_MAP must declare exactly the expected tuples (no missing, no extra).
    declared = _file_map_tuples()
    for src, tgt, deriv in sorted(expected - declared):
        problems.append(
            f"FILE_MAP is missing expected migration tuple ({src} -> {tgt}, {deriv}) "
            "(a derived target and/or its lineage row was dropped)"
        )
    for src, tgt, deriv in sorted(declared - expected):
        problems.append(
            f"FILE_MAP declares an unexpected migration tuple ({src} -> {tgt}, {deriv})"
        )

    # Every expected target must exist on disk (delete-both of a fan-out target).
    for _src, tgt, _deriv in sorted(expected):
        if not (DISS / tgt).is_file():
            problems.append(f"expected migration target missing on disk: {tgt}")

    return problems


def check_inventory_covers_labels() -> list[str]:
    problems: list[str] = []
    labels: set[str] = set()
    for tex in sorted(SRC_2026.glob("*.tex")):
        labels.update(_LABEL_RE.findall(tex.read_text(encoding="utf-8")))
    with INVENTORY.open(encoding="utf-8") as fh:
        inv_labels = {row["latex_label"] for row in csv.DictReader(fh)}
    for label in sorted(labels):
        if label not in inv_labels:
            problems.append(f"figure/table {label} in 2026 source has no inventory row")
    return problems


def check_file_map() -> list[str]:
    problems: list[str] = []

    with FILE_MAP.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != _EXPECTED_COLUMNS:
            problems.append(f"FILE_MAP columns {reader.fieldnames} != expected {_EXPECTED_COLUMNS}")
            return problems
        rows = list(reader)

    mapped: dict[str, dict[str, str]] = {}
    for row in rows:
        target_rel = row["dissertation_file"]
        if target_rel in mapped:
            problems.append(f"duplicate FILE_MAP row for {target_rel}")
            continue
        mapped[target_rel] = row

    expected = expected_derived_files()
    mapped_keys = set(mapped)
    for missing in sorted(expected - mapped_keys):
        problems.append(f"derived file has no FILE_MAP row (deletion or omission): {missing}")
    for extra in sorted(mapped_keys - expected):
        problems.append(f"FILE_MAP row for unexpected/undeclared target: {extra}")

    for target_rel, row in sorted(mapped.items()):
        target = DISS / target_rel
        if not target.is_file():
            problems.append(f"FILE_MAP references missing file: {target_rel}")
            continue
        if row["derivation"] not in _VALID_DERIVATIONS:
            problems.append(f"unknown derivation {row['derivation']!r} for {target_rel}")
            continue

        src = SRC_2026 / row["source_2026_file"]
        if not src.is_file():
            problems.append(f"missing 2026 source {row['source_2026_file']} for {target_rel}")
            continue
        if _sha256(src) != row["source_sha256"]:
            problems.append(f"2026 source hash drift: {row['source_2026_file']}")

        current = _sha256(target)
        if current != row["current_sha256"]:
            problems.append(
                f"current_sha256 stale for {target_rel} "
                f"(recorded {row['current_sha256'][:12]}..., actual {current[:12]}...)"
            )
        if (
            row["derivation"] == "byte-identical-copy"
            and row["current_sha256"] != row["source_sha256"]
        ):
            problems.append(f"byte-identical copy no longer matches source: {target_rel}")

    return problems


def main() -> int:
    problems = check_inventory_covers_labels() + check_migration_fanout() + check_file_map()
    if problems:
        print("Lineage check FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(
        "[check-lineage] inventory covers all 2026 figures/tables; the exact (source, target, "
        "derivation) migration fan-out matches FILE_MAP and the tree; FILE_MAP consistent."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
