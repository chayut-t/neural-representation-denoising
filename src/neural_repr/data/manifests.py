"""Split manifests, per-file hashes, and dataset validation.

Phase 4 (plan §7 tasks 4-5; decision 0001). A **manifest** is the committed record
of a dataset split: one row per image with its split ID, relative path, two digests,
and inspected properties (format, shape, color mode). The image *bytes* are never
committed — only the manifest (plan task 5). Manifests are CSV so a reviewer can read
them and a diff is meaningful.

Two digests, because they guarantee different things (see ``data/README.md`` and
``docs/tolerances.md``):

* ``content_sha256`` — a canonical digest over the **decoded pixels** (dtype, shape,
  raw bytes). PNG *decoding* is lossless and platform-independent, so this digest is
  reproducible on any platform. It is the reproducibility, split-leakage, and
  duplicate-detection invariant, and it is always present.
* ``sha256`` — the digest of the **file bytes as stored on disk**. PNG *encoding*
  (zlib) is not byte-identical across platforms or library builds even when the
  decoded pixels are identical, so file-byte identity is only a meaningful, checkable
  pin for files whose exact bytes we received and keep (downloaded third-party data).
  For the regenerable synthetic fixture — which is produced fresh on each platform —
  this column is intentionally **empty**: the fixture's contract is pixel identity,
  not encoded-byte identity. When present it is verified; when empty it is skipped.

Validation (plan task 4) checks image counts, IDs, formats, shapes, color modes,
corrupt/unreadable files, duplicate content, and **train/test split leakage** (the
same image content appearing in more than one split) — all keyed on the platform-
stable ``content_sha256``. These run against a local dataset directory and, for the
committed synthetic fixture, in CI.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from neural_repr.data.io_safe import atomic_write_text
from neural_repr.data.records import sha256_file

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_VALID_ROLES = frozenset({"train", "val", "test"})

_MANIFEST_COLUMNS = [
    "split_id",
    "role",
    "relative_path",
    "sha256",
    "content_sha256",
    "format",
    "height",
    "width",
    "mode",
]


@dataclass(frozen=True)
class ManifestRow:
    """One image's manifest entry.

    ``content_sha256`` (decoded-pixel digest) is always present; ``sha256`` (file-byte
    digest) may be an empty string for regenerable fixtures whose encoded bytes are not
    a cross-platform invariant. See the module docstring.
    """

    split_id: str
    role: str
    relative_path: str
    sha256: str
    content_sha256: str
    format: str
    height: int
    width: int
    mode: str


def image_content_digest(arr: NDArray[np.generic]) -> str:
    """Canonical SHA-256 over an image's decoded pixels (dtype, shape, raw bytes).

    Platform-independent (unlike PNG file bytes, whose zlib encoding varies by
    platform/library build), so it is the reproducibility and leakage invariant. The
    dtype and shape are folded in so arrays that share raw bytes but differ in layout
    cannot collide.
    """
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{tuple(int(x) for x in a.shape)}".encode())
    h.update(a.tobytes())
    return "sha256:" + h.hexdigest()


def decode_pixels(path: Path) -> NDArray[np.generic]:
    """Decode an image file to its pixel array (lossless for PNG; platform-stable)."""
    with Image.open(path) as im:
        im.load()
        return np.asarray(im)


def image_content_digest_file(path: Path) -> str:
    """Content digest of the decoded pixels of an image *file*."""
    return image_content_digest(decode_pixels(path))


def inspect_image(path: Path) -> tuple[str, int, int, str]:
    """Return ``(format, height, width, mode)`` for an image, or raise if unreadable."""
    with Image.open(path) as im:
        im.verify()  # detect truncated/corrupt files without full decode
    with Image.open(path) as im:
        width, height = im.size
        return (im.format or "UNKNOWN", int(height), int(width), im.mode)


def build_manifest_row(
    root: Path, path: Path, *, split_id: str, role: str, include_file_hash: bool = True
) -> ManifestRow:
    """Build one manifest row from an image file.

    Always records the platform-stable ``content_sha256``. Records the file-byte
    ``sha256`` too when ``include_file_hash`` (the default, for downloaded third-party
    files whose exact bytes are worth pinning); pass ``include_file_hash=False`` for
    regenerable fixtures whose encoded bytes are not a cross-platform invariant.
    """
    fmt, h, w, mode = inspect_image(path)
    return ManifestRow(
        split_id=split_id,
        role=role,
        relative_path=path.relative_to(root).as_posix(),
        sha256=sha256_file(path) if include_file_hash else "",
        content_sha256=image_content_digest_file(path),
        format=fmt,
        height=h,
        width=w,
        mode=mode,
    )


def manifest_to_csv(rows: list[ManifestRow]) -> str:
    """Serialize manifest rows to CSV text (sorted by role/split_id; stable + diffable)."""
    ordered = sorted(rows, key=lambda r: (r.role, r.split_id))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_MANIFEST_COLUMNS)
    writer.writeheader()
    for row in ordered:
        writer.writerow(asdict(row))
    return buf.getvalue()


def write_manifest(rows: list[ManifestRow], out_path: Path, *, overwrite: bool = False) -> None:
    """Write manifest rows to CSV, collision-failing by default (plan §0.3 no-overwrite)."""
    atomic_write_text(out_path, manifest_to_csv(rows), overwrite=overwrite)


def read_manifest(path: Path) -> list[ManifestRow]:
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != _MANIFEST_COLUMNS:
            raise ValueError(
                f"manifest columns {reader.fieldnames} != expected {_MANIFEST_COLUMNS}"
            )
        return [
            ManifestRow(
                split_id=r["split_id"],
                role=r["role"],
                relative_path=r["relative_path"],
                sha256=r["sha256"],
                content_sha256=r["content_sha256"],
                format=r["format"],
                height=int(r["height"]),
                width=int(r["width"]),
                mode=r["mode"],
            )
            for r in reader
        ]


def validate_manifest(
    rows: list[ManifestRow],
    *,
    expected_counts: dict[str, int] | None = None,
    allowed_formats: set[str] | None = None,
    allowed_modes: set[str] | None = None,
) -> list[str]:
    """Validate a manifest; return a list of problems (empty means valid).

    Checks (plan task 4): per-role counts (if expected given), duplicate split IDs,
    duplicate content hashes *across* splits (leakage) and within a split, allowed
    formats/color modes, and non-positive shapes. A pure function over manifest rows
    so it is unit-testable without a dataset on disk.

    Leakage and duplicate detection key on ``content_sha256`` (the platform-stable
    decoded-pixel digest), not the file-byte ``sha256`` — two encodings of the same
    pixels must count as the same content. The file-byte ``sha256`` is optional (empty
    for regenerable fixtures); it is only syntax-checked when present.
    """
    problems: list[str] = []

    # Counts per role.
    role_counts = Counter(r.role for r in rows)
    if expected_counts is not None:
        for role, expected in expected_counts.items():
            if role_counts.get(role, 0) != expected:
                problems.append(
                    f"role {role!r} has {role_counts.get(role, 0)} images, expected {expected}"
                )

    # Duplicate split IDs.
    id_counts = Counter(r.split_id for r in rows)
    for sid, n in id_counts.items():
        if n > 1:
            problems.append(f"duplicate split_id {sid!r} appears {n} times")

    # Per-row structural checks: role vocabulary, digest syntax, path safety, shapes.
    for r in rows:
        if r.role not in _VALID_ROLES:
            problems.append(f"{r.split_id}: role {r.role!r} not in {sorted(_VALID_ROLES)}")
        if not _SHA256_RE.match(r.content_sha256):
            problems.append(
                f"{r.split_id}: content_sha256 {r.content_sha256!r} is not 'sha256:'+64 "
                "lowercase hex"
            )
        # File-byte sha256 is optional (empty for regenerable fixtures); check syntax only
        # when a value is present.
        if r.sha256 and not _SHA256_RE.match(r.sha256):
            problems.append(f"{r.split_id}: sha256 {r.sha256!r} is not 'sha256:'+64 lowercase hex")
        if not _is_safe_relative_path(r.relative_path):
            problems.append(f"{r.split_id}: unsafe relative_path {r.relative_path!r}")
        if r.height <= 0 or r.width <= 0:
            problems.append(f"{r.split_id}: non-positive shape {r.height}x{r.width}")
        if allowed_formats is not None and r.format not in allowed_formats:
            problems.append(f"{r.split_id}: format {r.format!r} not in {sorted(allowed_formats)}")
        if allowed_modes is not None and r.mode not in allowed_modes:
            problems.append(f"{r.split_id}: color mode {r.mode!r} not in {sorted(allowed_modes)}")

    # Split leakage: identical decoded-pixel content in more than one role.
    by_hash: dict[str, set[str]] = {}
    for r in rows:
        by_hash.setdefault(r.content_sha256, set()).add(r.role)
    for digest, roles in by_hash.items():
        if len(roles) > 1:
            problems.append(
                f"split leakage: content {digest[:19]}... appears in roles {sorted(roles)}"
            )

    # Exact duplicate content within the dataset (even within one role) is worth flagging.
    hash_counts = Counter(r.content_sha256 for r in rows)
    for digest, n in hash_counts.items():
        if n > 1:
            problems.append(f"duplicate content hash {digest[:19]}... appears {n} times")

    return problems


def _is_safe_relative_path(rel: str) -> bool:
    """True iff ``rel`` is a normal relative path (no absolute, no ``..`` escape)."""
    if not rel or rel.startswith(("/", "\\")):
        return False
    p = Path(rel)
    if p.is_absolute():
        return False
    return ".." not in p.parts


def resolve_under_root(root: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``root``, raising if it escapes ``root`` (traversal guard)."""
    if not _is_safe_relative_path(rel):
        raise ValueError(f"unsafe relative path {rel!r}")
    root_resolved = root.resolve()
    target = (root_resolved / rel).resolve()
    if root_resolved not in target.parents and target != root_resolved:
        raise ValueError(f"path {rel!r} resolves outside root {root}")
    return target


def verify_manifest_on_disk(
    rows: list[ManifestRow],
    root: Path,
    *,
    expected_roles: set[str] | None = None,
    expected_ids: set[str] | None = None,
    expected_counts: dict[str, int] | None = None,
    allowed_formats: set[str] | None = None,
    allowed_modes: set[str] | None = None,
) -> list[str]:
    """Validate a manifest AND the actual bytes it points at (plan task 4; R2 finding 1).

    Runs :func:`validate_manifest` first (structure/leakage), then for every row
    resolves its ``relative_path`` under ``root`` with traversal protection, reopens
    and decodes the image (catching corrupt files), and compares the *actual* format,
    height, width, mode, decoded-pixel digest, and (when the row records one) file-byte
    SHA-256 to the recorded row. The decoded-pixel ``content_sha256`` is always
    verified; the file-byte ``sha256`` is verified only when present (it is empty for
    regenerable fixtures, whose encoded bytes are not a cross-platform invariant).
    Optionally enforces an exact set of expected roles / split IDs (from a registry
    spec). Returns a problem list.
    """
    problems = validate_manifest(
        rows,
        expected_counts=expected_counts,
        allowed_formats=allowed_formats,
        allowed_modes=allowed_modes,
    )

    if expected_roles is not None:
        seen_roles = {r.role for r in rows}
        for extra in sorted(seen_roles - expected_roles):
            problems.append(f"role {extra!r} not an expected role {sorted(expected_roles)}")
    if expected_ids is not None:
        seen_ids = {r.split_id for r in rows}
        for missing in sorted(expected_ids - seen_ids):
            problems.append(f"expected split_id {missing!r} missing from manifest")
        for extra in sorted(seen_ids - expected_ids):
            problems.append(f"unexpected split_id {extra!r} not in the expected set")

    for r in rows:
        try:
            path = resolve_under_root(root, r.relative_path)
        except ValueError as exc:
            problems.append(f"{r.split_id}: {exc}")
            continue
        if not path.is_file():
            problems.append(f"{r.split_id}: file missing on disk: {r.relative_path}")
            continue
        try:
            fmt, h, w, mode = inspect_image(path)
        except Exception as exc:
            problems.append(f"{r.split_id}: unreadable/corrupt image {r.relative_path} ({exc})")
            continue
        if (fmt, h, w, mode) != (r.format, r.height, r.width, r.mode):
            problems.append(
                f"{r.split_id}: metadata mismatch — disk {(fmt, h, w, mode)} != manifest "
                f"{(r.format, r.height, r.width, r.mode)}"
            )
        try:
            actual_content = image_content_digest_file(path)
        except Exception as exc:
            problems.append(f"{r.split_id}: unreadable/corrupt image {r.relative_path} ({exc})")
            continue
        if actual_content != r.content_sha256:
            problems.append(
                f"{r.split_id}: content hash mismatch — disk {actual_content[:19]}... != manifest "
                f"{r.content_sha256[:19]}..."
            )
        # File-byte identity is only checkable when the row pins it (downloaded data).
        if r.sha256:
            actual_file = sha256_file(path)
            if actual_file != r.sha256:
                problems.append(
                    f"{r.split_id}: file-byte hash mismatch — disk {actual_file[:19]}... != "
                    f"manifest {r.sha256[:19]}..."
                )
    return problems
