#!/usr/bin/env python3
"""Baseline-to-working-edition lineage completeness (plan §7 Phase 2 task 9/10; Gate P2).

Checks:
1. Inventory coverage — every figure/table label in the frozen 2026 source has a row in
   docs/rewrite-2026-inventory.csv (nothing dropped in the lineage).
2. Manifest-source coverage — anchored on the IMMUTABLE frozen-2026 MANIFEST.sha256: every
   required content source (chapters/abstract/appendix/bib/thesis driver) must be cited by at
   least one FILE_MAP row. Because the manifest cannot change, deleting a derived target AND its
   FILE_MAP row together is detected here (the tree-derived expected set in check 3 cannot see
   that case — both the file and its row are gone).
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

_LABEL_RE = re.compile(r"\\label\{((?:fig|tab):[^}]+)\}")
_VALID_DERIVATIONS = {"byte-identical-copy", "translated-from"}

# Content source files (in the frozen 2026 MANIFEST) that MUST be represented in the
# working edition. Anchored on the immutable manifest — NOT the mutable dissertation
# tree — so deleting a target file *and* its FILE_MAP row together is still detected.
# Excludes build scripts (doit.sh, .gitignore), figures (images/), and the built PDF.
_REQUIRED_SOURCE_BASENAMES = frozenset(
    {
        "abstract.tex",
        "appendix.tex",
        "chap1.tex",
        "chap2.tex",
        "chap3.tex",
        "chap4.tex",
        "chap5.tex",
        "references.bib",
        "thesis.tex",
    }
)
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


def _file_map_cited_sources() -> set[str]:
    """Set of ``source_2026_file`` basenames cited by any FILE_MAP row."""
    with FILE_MAP.open(encoding="utf-8") as fh:
        return {Path(row["source_2026_file"]).name for row in csv.DictReader(fh)}


def check_manifest_sources_are_mapped() -> list[str]:
    """Every required content source in the frozen manifest must be cited by FILE_MAP.

    This is the immutable-source-of-truth direction (independent of the current
    dissertation/ tree): if a derived target and its FILE_MAP row are BOTH deleted,
    the source it derived from is no longer cited here, so this check fails — closing
    the delete-both gap that a tree-derived expected-set cannot see.
    """
    problems: list[str] = []
    manifest_names = _manifest_source_files()
    missing_required = _REQUIRED_SOURCE_BASENAMES - manifest_names
    if missing_required:
        problems.append(
            f"frozen manifest is missing expected source files: {sorted(missing_required)} "
            "(manifest drift — the baseline itself changed)"
        )
    cited = _file_map_cited_sources()
    for src in sorted(_REQUIRED_SOURCE_BASENAMES & manifest_names):
        if src not in cited:
            problems.append(
                f"frozen-2026 source {src} is not cited by any FILE_MAP row "
                "(a derived file and its lineage row were dropped together)"
            )
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
    problems = (
        check_inventory_covers_labels() + check_manifest_sources_are_mapped() + check_file_map()
    )
    if problems:
        print("Lineage check FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(
        "[check-lineage] inventory covers all 2026 figures/tables; every frozen-manifest "
        "source is mapped; FILE_MAP complete + consistent."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
