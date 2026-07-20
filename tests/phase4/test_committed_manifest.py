"""The committed synthetic manifest must match freshly-regenerated fixture pixels.

Gate P4: re-running preprocessing yields identical manifests on *any* platform. The
synthetic fixture images are git-ignored (regenerable), but their manifest is
committed. The reproducibility contract is **decoded-pixel identity**, not PNG-encoded
byte identity — PNG (zlib) encoding is not byte-identical across platforms/library
builds even when the decoded pixels match, so the committed manifest records the
platform-stable ``content_sha256`` (digest over decoded pixels) and leaves the
file-byte ``sha256`` empty for this regenerable fixture.

This test regenerates the fixture in memory and asserts the per-image content digests
match the committed manifest exactly, so a drift in the generator is caught in CI on
every platform. See ``test_fixture_content_vs_encoding_determinism`` for the separation
of pixel determinism from file-encoding determinism.
"""

from __future__ import annotations

from pathlib import Path

from neural_repr.data.manifests import image_content_digest, read_manifest
from neural_repr.data.synthetic import synthetic_image

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_MANIFEST = REPO_ROOT / "data" / "manifests" / "synthetic.csv"

# Must match the committed fixture parameters (configs/dataset/synthetic.yaml,
# and the make-fixture defaults used to write data/manifests/synthetic.csv).
FIXTURE_N = 8
FIXTURE_SIZE = 32
FIXTURE_SEED = 0


def test_committed_synthetic_manifest_matches_regeneration() -> None:
    rows = read_manifest(COMMITTED_MANIFEST)
    assert len(rows) == FIXTURE_N, "committed manifest row count drifted from fixture params"

    by_path = {r.relative_path: r for r in rows}
    for i in range(FIXTURE_N):
        img = synthetic_image(i, size=FIXTURE_SIZE, seed=FIXTURE_SEED)
        rel = f"synth_{i:04d}.png"
        row = by_path[rel]
        # Decoded-pixel digest is the cross-platform reproducibility invariant.
        assert image_content_digest(img) == row.content_sha256, (
            f"regenerated {rel} content digest != committed manifest"
        )
        # The regenerable fixture pins no file-byte hash (PNG encoding is not a
        # cross-platform invariant); the column is intentionally empty.
        assert row.sha256 == "", f"synthetic fixture row {rel} should not pin a file-byte hash"
        assert (row.height, row.width, row.mode) == (FIXTURE_SIZE, FIXTURE_SIZE, "RGB")
