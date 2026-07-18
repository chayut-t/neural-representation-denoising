#!/usr/bin/env python3
"""Baseline-to-working-edition lineage completeness (plan §7 Phase 2 task 10; Gate P2).

Checks:
1. Every empirical/schematic figure and every table in the frozen 2026 source has a
   row in docs/rewrite-2026-inventory.csv (nothing dropped in the lineage).
2. Every file listed in dissertation/FILE_MAP.csv exists and, for byte-identical
   copies, still matches its recorded 2026 source hash.

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
INVENTORY = REPO_ROOT / "docs" / "rewrite-2026-inventory.csv"
FILE_MAP = REPO_ROOT / "dissertation" / "FILE_MAP.csv"

_LABEL_RE = re.compile(r"\\label\{((?:fig|tab):[^}]+)\}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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
        for row in csv.DictReader(fh):
            target = REPO_ROOT / "dissertation" / row["dissertation_file"]
            if not target.is_file():
                problems.append(f"FILE_MAP references missing file: {row['dissertation_file']}")
                continue
            if row["derivation"] == "byte-identical-copy":
                src = SRC_2026 / row["source_2026_file"]
                if not src.is_file():
                    problems.append(f"missing 2026 source: {row['source_2026_file']}")
                    continue
                if _sha256(src) != row["source_sha256"]:
                    problems.append(f"2026 source hash drift: {row['source_2026_file']}")
                if _sha256(target) != row["source_sha256"]:
                    problems.append(
                        f"byte-identical copy no longer matches source: {row['dissertation_file']}"
                    )
    return problems


def main() -> int:
    problems = check_inventory_covers_labels() + check_file_map()
    if problems:
        print("Lineage check FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("[check-lineage] inventory covers all 2026 figures/tables; FILE_MAP consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
