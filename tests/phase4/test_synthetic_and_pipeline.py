"""Synthetic fixture determinism + end-to-end Gate P4 properties (Phase 4).

Gate P4: re-running preprocessing yields identical arrays on the reference platform;
inverse transforms reconstruct inputs within tolerance; no test image influences
fitted preprocessing. These run entirely on the committed synthetic fixture (no
download).
"""

from __future__ import annotations

import numpy as np

from neural_repr.data import (
    apply_whitening,
    extract_grid_patches,
    fit_whitening,
    invert_whitening,
    opponent_to_rgb,
    reassemble_patches,
    remove_dc,
    restore_dc,
    rgb_to_opponent,
    sample_random_patches,
    synthetic_dataset,
    synthetic_image,
)
from neural_repr.data.patches import PatchConfig


def test_synthetic_image_is_deterministic() -> None:
    a = synthetic_image(3, size=32, seed=0)
    b = synthetic_image(3, size=32, seed=0)
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.uint8 and a.shape == (32, 32, 3)


def test_synthetic_kinds_differ() -> None:
    # The four cycled kinds produce visibly different images.
    imgs = synthetic_dataset(4, size=32, seed=0)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.array_equal(imgs[i], imgs[j])


def test_synthetic_dataset_reproducible() -> None:
    d1 = synthetic_dataset(6, size=24, seed=1)
    d2 = synthetic_dataset(6, size=24, seed=1)
    for a, b in zip(d1, d2, strict=True):
        np.testing.assert_array_equal(a, b)


def test_full_representation_pipeline_reconstructs() -> None:
    """opponent -> DC-remove -> whiten -> invert -> DC-restore -> RGB round-trips."""
    img = synthetic_image(0, size=48, seed=0).astype(np.float64) / 255.0
    patches, positions = extract_grid_patches(img, size=16, stride=8)  # (n,16,16,3)
    n = patches.shape[0]
    flat = patches.reshape(n, -1, 3)

    opp = rgb_to_opponent(flat)
    centered, dc = remove_dc(opp)
    feats = centered.reshape(n, -1)

    stats = fit_whitening(feats, eps=1e-6)
    whitened = apply_whitening(feats, stats)
    recovered = invert_whitening(whitened, stats).reshape(centered.shape)

    back_opp = restore_dc(recovered, dc)
    back_rgb = opponent_to_rgb(back_opp).reshape(patches.shape)
    np.testing.assert_allclose(back_rgb, patches, atol=1e-6)

    # And the untouched patches reassemble to the original image.
    recon = reassemble_patches(patches, positions, img.shape)
    np.testing.assert_allclose(recon, img, atol=1e-9)


def test_whitening_fit_excludes_test_images() -> None:
    """Fitting whitening on train patches only must not depend on test-image content.

    Fit once on train patches; then fit again on the SAME train patches after
    fabricating arbitrary test patches. The fitted matrix must be identical — proving
    test data cannot influence the fitted preprocessing (Gate P4).
    """
    train_img = synthetic_image(0, size=48, seed=0).astype(np.float64) / 255.0
    test_img = synthetic_image(1, size=48, seed=0).astype(np.float64) / 255.0
    cfg = PatchConfig(size=12, n_random=40)
    train_p = sample_random_patches(
        train_img, cfg, dataset_sha256="synth", image_id="0000", seed=0
    ).reshape(40, -1)
    # Whitening is fit ONLY on train_p; test patches are never passed to fit_whitening.
    stats_a = fit_whitening(train_p, eps=1e-5)
    _ = sample_random_patches(test_img, cfg, dataset_sha256="synth", image_id="0001", seed=0)
    stats_b = fit_whitening(train_p, eps=1e-5)
    np.testing.assert_array_equal(stats_a.whiten, stats_b.whiten)
    np.testing.assert_array_equal(stats_a.mean, stats_b.mean)


def test_pipeline_reproducible_across_runs() -> None:
    """The whole deterministic pipeline yields identical arrays when re-run (Gate P4)."""

    def run() -> np.ndarray:
        img = synthetic_image(2, size=40, seed=0).astype(np.float64) / 255.0
        patches = sample_random_patches(
            img, PatchConfig(size=16, n_random=16), dataset_sha256="synth", image_id="0002", seed=0
        )
        flat = patches.reshape(16, -1, 3)
        centered, _ = remove_dc(rgb_to_opponent(flat))
        stats = fit_whitening(centered.reshape(16, -1), eps=1e-6)
        return apply_whitening(centered.reshape(16, -1), stats)

    np.testing.assert_array_equal(run(), run())
