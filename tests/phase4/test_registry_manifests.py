"""Tests for the dataset registry, manifests, validation, and records (Phase 4 tasks 1-5)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from neural_repr.data import (
    DIV2K,
    ManifestRow,
    build_manifest_row,
    get_dataset,
    read_manifest,
    registered_datasets,
    synthetic_image,
    validate_manifest,
    write_manifest,
)
from neural_repr.data.records import (
    ArchiveHash,
    TermsAcceptance,
    read_archive_hashes,
    read_terms_acceptance,
    record_archive_hash,
    verify_archive,
    write_terms_acceptance,
)

# --- registry ---------------------------------------------------------------


def test_registry_has_div2k_and_bsds500() -> None:
    assert set(registered_datasets()) == {"div2k", "bsds500"}


def test_div2k_split_ranges() -> None:
    assert DIV2K.train_ids()[0] == 1
    assert DIV2K.train_ids()[-1] == 720
    assert len(DIV2K.train_ids()) == 720
    # Splits are disjoint.
    train = set(DIV2K.train_ids())
    val = {i for s in DIV2K.split_by_role("val") for i in s.ids()}
    test = {i for s in DIV2K.split_by_role("test") for i in s.ids()}
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)


def test_registry_unknown_dataset_raises() -> None:
    with pytest.raises(KeyError, match="unknown dataset"):
        get_dataset("imagenet")


def test_registry_records_terms_token() -> None:
    assert get_dataset("div2k").terms_acceptance_token == "--accept-academic-research-terms"
    assert get_dataset("bsds500").terms_acceptance_token == "--accept-noncommercial-research-terms"


# --- manifests + validation -------------------------------------------------


def _write_images(root: Path, specs: list[tuple[str, int]]) -> list[ManifestRow]:
    """Write synthetic PNGs and build manifest rows. specs = [(role, index), ...]."""
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (role, index) in enumerate(specs):
        img = synthetic_image(index, size=24, seed=0)
        rel = f"img_{i:03d}.png"
        Image.fromarray(img, mode="RGB").save(root / rel)
        rows.append(build_manifest_row(root, root / rel, split_id=f"{role}-{i:03d}", role=role))
    return rows


def test_manifest_roundtrip_csv(tmp_path: Path) -> None:
    rows = _write_images(tmp_path / "imgs", [("train", 0), ("train", 1)])
    out = tmp_path / "manifest.csv"
    write_manifest(rows, out)
    loaded = read_manifest(out)
    assert {r.split_id for r in loaded} == {r.split_id for r in rows}
    assert all(r.mode == "RGB" and r.format == "PNG" for r in loaded)


def test_validate_clean_manifest(tmp_path: Path) -> None:
    rows = _write_images(tmp_path / "imgs", [("train", 0), ("train", 1), ("test", 2)])
    assert validate_manifest(rows) == []


def test_validate_detects_split_leakage(tmp_path: Path) -> None:
    # Same image content (same index) placed in train and test -> identical hash.
    root = tmp_path / "imgs"
    root.mkdir(parents=True)
    img = synthetic_image(0, size=24, seed=0)
    from PIL import Image as _Image

    _Image.fromarray(img, mode="RGB").save(root / "a.png")
    _Image.fromarray(img, mode="RGB").save(root / "b.png")
    rows = [
        build_manifest_row(root, root / "a.png", split_id="train-0", role="train"),
        build_manifest_row(root, root / "b.png", split_id="test-0", role="test"),
    ]
    problems = validate_manifest(rows)
    assert any("split leakage" in p for p in problems)


def test_validate_detects_duplicate_ids() -> None:
    row = ManifestRow("dup", "train", "a.png", "sha256:" + "0" * 64, "PNG", 8, 8, "RGB")
    row2 = ManifestRow("dup", "train", "b.png", "sha256:" + "1" * 64, "PNG", 8, 8, "RGB")
    assert any("duplicate split_id" in p for p in validate_manifest([row, row2]))


def test_validate_expected_counts_and_modes() -> None:
    rows = [
        ManifestRow("a", "train", "a.png", "sha256:" + "0" * 64, "PNG", 8, 8, "RGB"),
        ManifestRow("b", "train", "b.png", "sha256:" + "1" * 64, "JPEG", 8, 8, "L"),
    ]
    problems = validate_manifest(
        rows, expected_counts={"train": 3}, allowed_formats={"PNG"}, allowed_modes={"RGB"}
    )
    assert any("expected 3" in p for p in problems)
    assert any("format" in p for p in problems)
    assert any("color mode" in p for p in problems)


def test_inspect_corrupt_image_raises(tmp_path: Path) -> None:
    from neural_repr.data import inspect_image

    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a real png")
    with pytest.raises(Exception):  # noqa: B017 - PIL raises various types on corrupt input
        inspect_image(bad)


def test_validate_rejects_bad_role_and_digest_and_path() -> None:
    rows = [
        ManifestRow("a", "unknown-role", "../../x.png", "not-a-digest", "PNG", 8, 8, "RGB"),
    ]
    problems = validate_manifest(rows)
    assert any("role" in p for p in problems)
    assert any("sha256" in p for p in problems)
    assert any("unsafe relative_path" in p for p in problems)


def test_verify_on_disk_detects_byte_change_after_manifest(tmp_path: Path) -> None:
    from neural_repr.data import verify_manifest_on_disk

    root = tmp_path / "imgs"
    rows = _write_images(root, [("train", 0), ("train", 1)])
    # Clean: on-disk verification passes.
    assert verify_manifest_on_disk(rows, root) == []
    # Mutate the bytes AFTER the manifest was built -> hash mismatch detected.
    from PIL import Image as _Image

    _Image.fromarray(synthetic_image(5, size=24, seed=0), mode="RGB").save(root / "img_000.png")
    problems = verify_manifest_on_disk(rows, root)
    assert any("content hash mismatch" in p for p in problems)


def test_verify_on_disk_detects_corrupt_image(tmp_path: Path) -> None:
    from neural_repr.data import verify_manifest_on_disk

    root = tmp_path / "imgs"
    rows = _write_images(root, [("train", 0)])
    (root / "img_000.png").write_bytes(b"corrupted")
    problems = verify_manifest_on_disk(rows, root)
    assert any("corrupt" in p or "hash mismatch" in p for p in problems)


def test_verify_on_disk_detects_missing_file(tmp_path: Path) -> None:
    from neural_repr.data import verify_manifest_on_disk

    root = tmp_path / "imgs"
    rows = _write_images(root, [("train", 0)])
    (root / "img_000.png").unlink()
    assert any("file missing" in p for p in verify_manifest_on_disk(rows, root))


def test_resolve_under_root_blocks_traversal(tmp_path: Path) -> None:
    from neural_repr.data import resolve_under_root

    with pytest.raises(ValueError, match="unsafe relative path"):
        resolve_under_root(tmp_path, "../escape.png")
    ok = resolve_under_root(tmp_path, "sub/a.png")
    assert ok == (tmp_path / "sub" / "a.png").resolve()


def test_verify_on_disk_enforces_expected_roles(tmp_path: Path) -> None:
    from neural_repr.data import verify_manifest_on_disk

    root = tmp_path / "imgs"
    rows = _write_images(root, [("train", 0), ("test", 1)])
    # Only 'train' expected -> the 'test' row is flagged.
    problems = verify_manifest_on_disk(rows, root, expected_roles={"train"})
    assert any("not an expected role" in p for p in problems)


# --- local records (git-ignored) --------------------------------------------


def test_terms_acceptance_roundtrip(tmp_path: Path) -> None:
    acc = TermsAcceptance(
        "div2k", "https://example/terms", "--accept-x", "sha256:" + "e" * 64, "2026-07-19T00:00:00Z"
    )
    write_terms_acceptance(tmp_path, acc)
    loaded = read_terms_acceptance(tmp_path, "div2k")
    assert loaded == acc
    assert loaded.terms_sha256 == "sha256:" + "e" * 64
    assert read_terms_acceptance(tmp_path, "bsds500") is None


def _accept(records: Path, dataset: str = "div2k") -> None:
    """Write a terms-acceptance record so archive pinning is permitted."""
    write_terms_acceptance(
        records,
        TermsAcceptance(
            dataset,
            "https://example/terms",
            "--accept-x",
            "sha256:" + "f" * 64,
            "2026-07-19T00:00:00Z",
        ),
    )


def test_archive_hash_record_is_create_only_and_verifies(tmp_path: Path) -> None:
    records = tmp_path / "records"
    _accept(records)
    archive = tmp_path / "DIV2K_train_HR.zip"
    archive.write_bytes(b"pretend archive bytes")
    from neural_repr.data.records import sha256_file

    record_archive_hash(records, ArchiveHash("div2k", "DIV2K_train_HR.zip", sha256_file(archive)))
    assert "DIV2K_train_HR.zip" in read_archive_hashes(records, "div2k")
    verify_archive(records, "div2k", "DIV2K_train_HR.zip", archive)

    # Re-recording the SAME digest is idempotent.
    record_archive_hash(records, ArchiveHash("div2k", "DIV2K_train_HR.zip", sha256_file(archive)))

    # A DIFFERENT digest for the same archive is refused (immutable pin).
    from neural_repr.data.records import ArchivePinConflict

    with pytest.raises(ArchivePinConflict, match="refusing to change"):
        record_archive_hash(
            records, ArchiveHash("div2k", "DIV2K_train_HR.zip", "sha256:" + "0" * 64)
        )

    # Changed bytes still fail verification against the original pin.
    archive.write_bytes(b"tampered bytes")
    with pytest.raises(ValueError, match="mismatch"):
        verify_archive(records, "div2k", "DIV2K_train_HR.zip", archive)


def test_pin_requires_terms_acceptance(tmp_path: Path) -> None:
    records = tmp_path / "records"  # no acceptance written
    with pytest.raises(ValueError, match="no terms-acceptance record"):
        record_archive_hash(
            records, ArchiveHash("div2k", "DIV2K_train_HR.zip", "sha256:" + "1" * 64)
        )


def test_rotation_is_explicit_and_logged(tmp_path: Path) -> None:
    from neural_repr.data.records import rotate_archive_hash

    records = tmp_path / "records"
    _accept(records)
    record_archive_hash(records, ArchiveHash("div2k", "DIV2K_train_HR.zip", "sha256:" + "1" * 64))
    rotate_archive_hash(
        records,
        ArchiveHash("div2k", "DIV2K_train_HR.zip", "sha256:" + "2" * 64),
        reason="official re-release",
    )
    assert read_archive_hashes(records, "div2k")["DIV2K_train_HR.zip"] == "sha256:" + "2" * 64
    assert (records / "div2k.rotation-log.jsonl").is_file()


def test_verify_without_record_raises(tmp_path: Path) -> None:
    archive = tmp_path / "DIV2K_train_HR.zip"
    archive.write_bytes(b"x")
    with pytest.raises(ValueError, match="no recorded hash"):
        verify_archive(tmp_path / "records", "div2k", "DIV2K_train_HR.zip", archive)
