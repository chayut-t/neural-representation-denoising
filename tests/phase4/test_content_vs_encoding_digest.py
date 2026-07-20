"""Regression: separate pixel-content determinism from PNG-encoding determinism.

The CI failure this guards against (2026-07-20): the committed synthetic manifest
pinned PNG *file-byte* hashes, which are not identical across platforms/library builds
because zlib encoding differs even when the decoded pixels are identical. The fixture's
real reproducibility contract is decoded-pixel identity, captured by ``content_sha256``.

These tests assert the two properties independently, so a future regression is
attributed correctly:

* pixel content is deterministic and platform-independent (must always hold);
* PNG file bytes are *not* part of the contract (round-tripping decoded pixels yields
  the same content digest even if the re-encoded bytes differ).
"""

from __future__ import annotations

import io
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from neural_repr.data.manifests import (
    build_manifest_row,
    decode_pixels,
    image_content_digest,
    image_content_digest_file,
)
from neural_repr.data.synthetic import synthetic_image


def test_content_digest_is_deterministic_across_regeneration() -> None:
    """Regenerating the same (seed, index, size) yields the same content digest."""
    a = synthetic_image(3, size=32, seed=0)
    b = synthetic_image(3, size=32, seed=0)
    assert image_content_digest(a) == image_content_digest(b)


def test_content_digest_survives_png_roundtrip() -> None:
    """PNG decode is lossless: the content digest is stable across an encode/decode."""
    img = synthetic_image(1, size=32, seed=0)
    buf = io.BytesIO()
    Image.fromarray(img, mode="RGB").save(buf, format="PNG")
    decoded = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    assert image_content_digest(decoded) == image_content_digest(img)


def test_content_digest_independent_of_png_encoding_choices(tmp_path: Path) -> None:
    """Same pixels, different PNG *file bytes* -> same content digest, different file hash.

    Re-encoding at a different zlib compression level produces different file bytes
    (mimicking cross-platform encoder differences) while the decoded pixels — and hence
    the content digest — are unchanged. This is exactly the situation that broke CI.
    """
    from neural_repr.data.records import sha256_file

    img = synthetic_image(0, size=32, seed=0)
    p0 = tmp_path / "level0.png"
    p9 = tmp_path / "level9.png"
    Image.fromarray(img, mode="RGB").save(p0, format="PNG", compress_level=0)
    Image.fromarray(img, mode="RGB").save(p9, format="PNG", compress_level=9)

    # Different encoded bytes (the fragile property we no longer rely on)...
    assert sha256_file(p0) != sha256_file(p9)
    # ...but identical decoded-pixel content (the property the manifest pins).
    assert image_content_digest_file(p0) == image_content_digest_file(p9)
    assert (decode_pixels(p0) == decode_pixels(p9)).all()


def test_build_row_records_content_digest_and_optional_file_hash(tmp_path: Path) -> None:
    """The synthetic-fixture path omits the file-byte hash; the default keeps it."""
    img = synthetic_image(0, size=32, seed=0)
    path = tmp_path / "img.png"
    Image.fromarray(img, mode="RGB").save(path, format="PNG")

    fixture_row = build_manifest_row(
        tmp_path, path, split_id="s-0", role="train", include_file_hash=False
    )
    assert fixture_row.sha256 == ""
    assert fixture_row.content_sha256 == image_content_digest(img)

    pinned_row = build_manifest_row(tmp_path, path, split_id="s-0", role="train")
    assert pinned_row.sha256.startswith("sha256:")
    assert pinned_row.content_sha256 == fixture_row.content_sha256


def test_manifest_content_digest_matches_raw_zlib_of_pixels() -> None:
    """Sanity check the digest domain: it is over raw pixel bytes, not compressed bytes."""
    img = synthetic_image(2, size=16, seed=0)
    # The digest folds dtype+shape then raw bytes; it must not equal a digest over
    # zlib-compressed bytes (guards against accidentally hashing an encoded stream).
    compressed = zlib.compress(np.ascontiguousarray(img).tobytes())
    import hashlib

    assert image_content_digest(img) != "sha256:" + hashlib.sha256(compressed).hexdigest()
