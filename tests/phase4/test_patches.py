"""Tests for deterministic patch sampling, augmentation, reassembly (Phase 4 tasks 6-9)."""

from __future__ import annotations

import numpy as np
import pytest

from neural_repr.data import (
    AUGMENTATIONS,
    PatchConfig,
    apply_augmentation,
    extract_grid_patches,
    reassemble_patches,
    sample_random_patches,
    synthetic_image,
)
from neural_repr.data.patches import _patch_seed

RNG = np.random.default_rng(0)


def _image(size: int = 48) -> np.ndarray:
    return synthetic_image(0, size=size, seed=0).astype(np.float64) / 255.0


def test_patch_seed_is_stable_and_content_dependent() -> None:
    s1 = _patch_seed("abc", "0001", 0, 3)
    assert s1 == _patch_seed("abc", "0001", 0, 3)  # stable
    assert s1 != _patch_seed("abc", "0001", 0, 4)  # patch index matters
    assert s1 != _patch_seed("abc", "0002", 0, 3)  # image id matters
    assert s1 != _patch_seed("xyz", "0001", 0, 3)  # dataset hash matters
    assert s1 != _patch_seed("abc", "0001", 1, 3)  # seed matters


def test_random_patches_deterministic() -> None:
    img = _image()
    cfg = PatchConfig(size=16, n_random=10)
    p1 = sample_random_patches(img, cfg, dataset_sha256="d", image_id="0001", seed=0)
    p2 = sample_random_patches(img, cfg, dataset_sha256="d", image_id="0001", seed=0)
    np.testing.assert_array_equal(p1, p2)
    assert p1.shape == (10, 16, 16, 3)


def test_random_patches_change_with_seed() -> None:
    img = _image()
    cfg = PatchConfig(size=16, n_random=10)
    p1 = sample_random_patches(img, cfg, dataset_sha256="d", image_id="0001", seed=0)
    p2 = sample_random_patches(img, cfg, dataset_sha256="d", image_id="0001", seed=1)
    assert not np.array_equal(p1, p2)


def test_random_patches_respect_border() -> None:
    img = _image(size=64)
    cfg = PatchConfig(size=16, n_random=50, border=8)
    patches = sample_random_patches(img, cfg, dataset_sha256="d", image_id="x", seed=0)
    assert patches.shape == (50, 16, 16, 3)  # all fit within the border


def test_random_patches_too_small_raises() -> None:
    with pytest.raises(ValueError, match="too small"):
        sample_random_patches(
            _image(size=16),
            PatchConfig(size=16, n_random=1, border=4),
            dataset_sha256="d",
            image_id="x",
            seed=0,
        )


@pytest.mark.parametrize("name", AUGMENTATIONS)
def test_augmentations_preserve_shape_and_content(name: str) -> None:
    patch = RNG.uniform(0, 1, (5, 5, 3))
    out = apply_augmentation(patch, name)
    assert out.shape == patch.shape
    # Dihedral ops are permutations of pixels: same multiset of values.
    np.testing.assert_allclose(np.sort(out.ravel()), np.sort(patch.ravel()))


def test_augmentation_rot90_four_times_identity() -> None:
    patch = RNG.uniform(0, 1, (7, 7, 3))
    out = patch
    for _ in range(4):
        out = apply_augmentation(out, "rot90")
    np.testing.assert_allclose(out, patch)


def test_augmentation_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown augmentation"):
        apply_augmentation(np.zeros((3, 3, 3)), "color_jitter")


def test_grid_patches_cover_and_reassemble_exactly() -> None:
    img = _image(size=50)  # not a multiple of stride -> exercises last-position fixup
    patches, positions = extract_grid_patches(img, size=16, stride=8)
    recon = reassemble_patches(patches, positions, img.shape)
    np.testing.assert_allclose(recon, img, atol=1e-9)


def test_reassemble_box_window_also_reconstructs() -> None:
    img = _image(size=48)
    patches, positions = extract_grid_patches(img, size=16, stride=16)  # non-overlapping
    recon = reassemble_patches(patches, positions, img.shape, blend=False)
    np.testing.assert_allclose(recon, img, atol=1e-9)


def test_patch_config_validation() -> None:
    with pytest.raises(ValueError, match="size"):
        PatchConfig(size=0)
    with pytest.raises(ValueError, match="stride"):
        PatchConfig(stride=0)
    with pytest.raises(ValueError, match="border"):
        PatchConfig(border=-1)
