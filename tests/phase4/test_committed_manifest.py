"""The committed synthetic manifest must match freshly-regenerated fixture bytes.

Gate P4: re-running preprocessing yields identical manifests on the reference
platform. The synthetic fixture images are git-ignored (regenerable), but their
manifest is committed; this test regenerates the fixture in a tmp dir and asserts the
per-file SHA-256 hashes match the committed manifest exactly — so a drift in the
generator (or a non-deterministic PNG encode) is caught in CI.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from neural_repr.data.manifests import read_manifest
from neural_repr.data.records import sha256_file
from neural_repr.data.synthetic import synthetic_image

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_MANIFEST = REPO_ROOT / "data" / "manifests" / "synthetic.csv"

# Must match the committed fixture parameters (configs/dataset/synthetic.yaml,
# and the make-fixture defaults used to write data/manifests/synthetic.csv).
FIXTURE_N = 8
FIXTURE_SIZE = 32
FIXTURE_SEED = 0


def test_committed_synthetic_manifest_matches_regeneration(tmp_path: Path) -> None:
    rows = read_manifest(COMMITTED_MANIFEST)
    assert len(rows) == FIXTURE_N, "committed manifest row count drifted from fixture params"

    by_path = {r.relative_path: r for r in rows}
    for i in range(FIXTURE_N):
        img = synthetic_image(i, size=FIXTURE_SIZE, seed=FIXTURE_SEED)
        rel = f"synth_{i:04d}.png"
        out = tmp_path / rel
        Image.fromarray(img, mode="RGB").save(out)
        row = by_path[rel]
        assert sha256_file(out) == row.sha256, f"regenerated {rel} hash != committed manifest"
        assert (row.height, row.width, row.mode) == (FIXTURE_SIZE, FIXTURE_SIZE, "RGB")
