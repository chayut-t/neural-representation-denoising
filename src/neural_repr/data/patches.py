"""Deterministic patch sampling, augmentation, and overlapping reassembly.

Phase 4 (plan §7 tasks 6-9). Patch sampling is **deterministic from content, not
iteration order**: each patch's location is drawn from a generator seeded by
``(dataset_sha256, image_id, seed, patch_index)`` (plan task 6), so re-running
yields identical patches on any platform and the sample is independent of how images
happen to be enumerated.

Augmentation is restricted to the dihedral group of flips and 90-degree rotations
(plan task 8) — never color jitter, which would corrupt the color-dependence study.

Reassembly of overlapping patches uses a fixed, separable raised-cosine (Hann)
blending window normalized by the sum of overlapping windows, so a full image is
reconstructed without seams and — with no processing between extract and reassemble
— round-trips to the original within tolerance (Gate P4).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# The 8 dihedral augmentations (identity + flips + 90-degree rotations). Named so a
# config/record can pin an exact op rather than a random one.
AUGMENTATIONS: tuple[str, ...] = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "transpose",
    "anti_transpose",
)


@dataclass(frozen=True)
class PatchConfig:
    """Configurable patch-extraction policy (plan task 7).

    ``size`` is the square patch side; ``stride`` the step for the regular grid used
    by reassembly; ``n_random`` the number of randomly located patches to sample per
    image; ``border`` a margin (pixels) excluded from random sampling; ``augment``
    whether to apply a deterministically chosen dihedral op per sampled patch.
    """

    size: int = 16
    stride: int = 8
    n_random: int = 64
    border: int = 0
    augment: bool = False

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("patch size must be >= 1")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if self.n_random < 0:
            raise ValueError("n_random must be >= 0")
        if self.border < 0:
            raise ValueError("border must be >= 0")


def _patch_seed(dataset_sha256: str, image_id: str, seed: int, patch_index: int) -> int:
    """A stable 64-bit seed from ``(dataset hash, image id, seed, patch index)``.

    Hashing (not Python's salted ``hash``) so the seed is identical across processes
    and platforms — the determinism guarantee of plan task 6.
    """
    key = f"{dataset_sha256}|{image_id}|{seed}|{patch_index}".encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big")


def apply_augmentation(patch: Array, name: str) -> Array:
    """Apply a named dihedral augmentation to a channel-last ``(H, W, C)`` patch."""
    if name == "identity":
        return patch
    if name == "rot90":
        return np.rot90(patch, k=1, axes=(0, 1))
    if name == "rot180":
        return np.rot90(patch, k=2, axes=(0, 1))
    if name == "rot270":
        return np.rot90(patch, k=3, axes=(0, 1))
    if name == "flip_h":
        return patch[:, ::-1, :]
    if name == "flip_v":
        return patch[::-1, :, :]
    if name == "transpose":
        return np.swapaxes(patch, 0, 1)
    if name == "anti_transpose":
        return np.rot90(np.swapaxes(patch, 0, 1), k=2, axes=(0, 1))
    raise ValueError(f"unknown augmentation {name!r}; expected one of {AUGMENTATIONS}")


def sample_random_patches(
    image: Array,
    config: PatchConfig,
    *,
    dataset_sha256: str,
    image_id: str,
    seed: int,
) -> Array:
    """Deterministically sample ``config.n_random`` patches from a channel-last image.

    Returns an array of shape ``(n_random, size, size, C)``. Locations (and the
    augmentation choice, if enabled) are drawn from a per-patch generator seeded by
    ``(dataset_sha256, image_id, seed, patch_index)`` so the sample is reproducible
    and order-independent.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 3:
        raise ValueError(f"image must be (H, W, C), got shape {image.shape}")
    h, w, c = image.shape
    size, border = config.size, config.border
    max_top = h - size - border
    max_left = w - size - border
    if max_top < border or max_left < border:
        raise ValueError(f"image {h}x{w} too small for size {size} with border {border}")

    patches = np.empty((config.n_random, size, size, c), dtype=np.float64)
    for i in range(config.n_random):
        rng = np.random.default_rng(_patch_seed(dataset_sha256, image_id, seed, i))
        top = int(rng.integers(border, max_top + 1))
        left = int(rng.integers(border, max_left + 1))
        patch = image[top : top + size, left : left + size, :]
        if config.augment:
            aug = AUGMENTATIONS[int(rng.integers(0, len(AUGMENTATIONS)))]
            patch = apply_augmentation(patch, aug)
        patches[i] = patch
    return patches


def _grid_positions(length: int, size: int, stride: int) -> list[int]:
    """Top/left positions of a regular grid that always covers the last pixel."""
    if length < size:
        raise ValueError(f"dimension {length} smaller than patch size {size}")
    positions = list(range(0, length - size + 1, stride))
    last = length - size
    if positions[-1] != last:
        positions.append(last)
    return positions


def extract_grid_patches(
    image: Array, size: int, stride: int
) -> tuple[Array, list[tuple[int, int]]]:
    """Extract overlapping patches on a regular grid covering the whole image.

    Returns ``(patches, positions)`` where ``patches`` is ``(n, size, size, C)`` and
    ``positions`` are the ``(top, left)`` corners, for exact reassembly.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 3:
        raise ValueError(f"image must be (H, W, C), got shape {image.shape}")
    h, w, _ = image.shape
    tops = _grid_positions(h, size, stride)
    lefts = _grid_positions(w, size, stride)
    positions = [(t, ln) for t in tops for ln in lefts]
    patches = np.stack([image[t : t + size, ln : ln + size, :] for t, ln in positions])
    return patches, positions


def _hann_window_2d(size: int) -> Array:
    """Separable raised-cosine (Hann) window, strictly positive on the interior.

    A periodic Hann is 0 at the first sample; we use the symmetric form and floor it
    so every pixel gets nonzero weight, keeping the overlap-normalized reassembly
    well-defined even at patch edges.
    """
    n = np.arange(size)
    w1d = 0.5 - 0.5 * np.cos(2.0 * np.pi * (n + 1) / (size + 1))
    w1d = np.maximum(w1d, 1e-3)
    return np.outer(w1d, w1d)


def reassemble_patches(
    patches: Array,
    positions: list[tuple[int, int]],
    image_shape: tuple[int, int, int],
    *,
    blend: bool = True,
) -> Array:
    """Reassemble overlapping patches into a full image with normalized blending.

    Each patch is accumulated weighted by a fixed Hann window (or a box window if
    ``blend=False``); the accumulator is divided by the summed weights, so
    overlapping regions are averaged seamlessly. With patches taken straight from
    :func:`extract_grid_patches` (no processing between), this reconstructs the
    original image within floating-point tolerance.
    """
    h, w, c = image_shape
    size = patches.shape[1]
    window = _hann_window_2d(size) if blend else np.ones((size, size))
    window3 = window[:, :, None]
    acc = np.zeros((h, w, c), dtype=np.float64)
    wsum = np.zeros((h, w, 1), dtype=np.float64)
    for patch, (top, left) in zip(patches, positions, strict=True):
        acc[top : top + size, left : left + size, :] += patch * window3
        wsum[top : top + size, left : left + size, :] += window[:, :, None]
    if np.any(wsum == 0.0):
        raise ValueError("reassembly leaves uncovered pixels; check stride/positions")
    return acc / wsum
