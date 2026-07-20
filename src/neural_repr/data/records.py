"""Local, non-committed records: terms acceptance and archive hashes.

Phase 4 (plan §7 tasks 1, 3; decision 0001). These records are **personal and
local** — they capture that *this operator* accepted a dataset's terms at a time,
and the SHA-256 of the archive bytes they downloaded. The plan requires that
personal acceptance records are NOT committed (plan task 3), so they are written
under ``data/raw`` / ``data/interim`` (git-ignored) rather than into the tree.

The archive-hash record implements the "pin then verify" policy (§3.2): on first
authenticated download the operator records the archive hash here; every later run
verifies the downloaded bytes against it and fails loudly on a mismatch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TermsAcceptance:
    """A local record that the operator accepted a dataset's terms.

    ``terms_sha256`` is the SHA-256 of the exact terms text the operator captured at
    acceptance time (Phase 4 task 3): a URL alone is not an immutable record of which
    terms were agreed to. ``accepted_at_iso`` is a caller-supplied ISO-8601 timestamp.
    """

    dataset: str
    terms_url: str
    terms_acceptance_token: str
    terms_sha256: str
    accepted_at_iso: str  # caller supplies the timestamp (kept out of committed code)


@dataclass(frozen=True)
class ArchiveHash:
    """A recorded SHA-256 for one downloaded official archive."""

    dataset: str
    archive: str
    sha256: str


def _acceptance_path(records_dir: Path, dataset: str) -> Path:
    return records_dir / f"{dataset}.terms-acceptance.json"


def _archive_hash_path(records_dir: Path, dataset: str) -> Path:
    return records_dir / f"{dataset}.archive-hashes.json"


def write_terms_acceptance(records_dir: Path, acceptance: TermsAcceptance) -> Path:
    """Write a terms-acceptance record to the (git-ignored) records directory."""
    records_dir.mkdir(parents=True, exist_ok=True)
    path = _acceptance_path(records_dir, acceptance.dataset)
    path.write_text(json.dumps(asdict(acceptance), indent=2, sort_keys=True) + "\n")
    return path


def read_terms_acceptance(records_dir: Path, dataset: str) -> TermsAcceptance | None:
    path = _acceptance_path(records_dir, dataset)
    if not path.is_file():
        return None
    return TermsAcceptance(**json.loads(path.read_text()))


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


class ArchivePinConflict(ValueError):
    """Raised when recording an archive hash would change an existing pin."""


def record_archive_hash(
    records_dir: Path, item: ArchiveHash, *, require_acceptance: bool = True
) -> Path:
    """Record one archive's hash **create-only** (first-download pinning; decision 0001).

    A pin is immutable: re-recording the *same* digest is idempotent, but a
    *different* digest for an already-pinned archive raises :class:`ArchivePinConflict`
    rather than silently replacing the trusted pin (rotation is a separate, audited
    step — see :func:`rotate_archive_hash`). ``require_acceptance`` (default) refuses
    to pin unless a terms-acceptance record for the same dataset already exists, so an
    archive cannot be trusted without recorded terms agreement.
    """
    if require_acceptance and read_terms_acceptance(records_dir, item.dataset) is None:
        raise ValueError(
            f"cannot pin {item.dataset}/{item.archive}: no terms-acceptance record. "
            "Run accept-terms first (decision 0001)."
        )
    records_dir.mkdir(parents=True, exist_ok=True)
    path = _archive_hash_path(records_dir, item.dataset)
    existing: dict[str, str] = {}
    if path.is_file():
        existing = json.loads(path.read_text())
    if item.archive in existing:
        if existing[item.archive] == item.sha256:
            return path  # idempotent: same pin
        raise ArchivePinConflict(
            f"refusing to change existing pin for {item.dataset}/{item.archive}: "
            f"recorded {existing[item.archive]}, got {item.sha256}. Use an explicit "
            "audited rotation (rotate_archive_hash) if the official archive legitimately changed."
        )
    existing[item.archive] = item.sha256
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    return path


def rotate_archive_hash(records_dir: Path, item: ArchiveHash, *, reason: str) -> Path:
    """Explicitly rotate an existing archive pin to a new digest (audited step).

    Separate from :func:`record_archive_hash` so a pin change is always deliberate and
    leaves a reason on record. Appends to a rotation log alongside the hash record.
    """
    if not reason.strip():
        raise ValueError("rotation requires a non-empty reason")
    records_dir.mkdir(parents=True, exist_ok=True)
    path = _archive_hash_path(records_dir, item.dataset)
    existing: dict[str, str] = json.loads(path.read_text()) if path.is_file() else {}
    prior = existing.get(item.archive)
    existing[item.archive] = item.sha256
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    log = records_dir / f"{item.dataset}.rotation-log.jsonl"
    with log.open("a") as fh:
        fh.write(
            json.dumps(
                {"archive": item.archive, "from": prior, "to": item.sha256, "reason": reason}
            )
            + "\n"
        )
    return path


def read_archive_hashes(records_dir: Path, dataset: str) -> dict[str, str]:
    path = _archive_hash_path(records_dir, dataset)
    if not path.is_file():
        return {}
    result: dict[str, str] = json.loads(path.read_text())
    return result


def verify_archive(records_dir: Path, dataset: str, archive: str, path: Path) -> None:
    """Verify a downloaded archive against the recorded hash; raise on mismatch/absence.

    Implements the §3.2 "fail with an explanatory message rather than silently
    accepting new bytes" rule. A missing record is itself an error — the operator
    must record the hash (on a trusted first download) before verification can run.
    """
    recorded = read_archive_hashes(records_dir, dataset).get(archive)
    if recorded is None:
        raise ValueError(
            f"no recorded hash for {dataset}/{archive}; record it from a trusted "
            "download before verifying (see data/README.md)"
        )
    actual = sha256_file(path)
    if actual != recorded:
        raise ValueError(
            f"archive hash mismatch for {dataset}/{archive}: recorded {recorded}, "
            f"got {actual} — refusing to accept changed bytes"
        )
