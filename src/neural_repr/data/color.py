"""Color transforms: sRGB gamma, opponent-color basis, and DC removal.

Phase 4 (plan §7 task 9; decision 0003). All transforms here are pure NumPy and
**invertible within tolerance** — the inverse round-trips to the input so a
reconstructed image can be compared against the original (Gate P4). Conventions are
documented explicitly because the 2016/2026 baseline flags calibration confusion
(§3.7.1) from leaving the sRGB-vs-linear question implicit.

Array conventions
-----------------
* Images/patches are float64 arrays with the color channel **last**: shape
  ``(..., 3)`` for RGB/opponent data (so ``(H, W, 3)`` images and ``(N, P, 3)``
  patch stacks both work). Values are in ``[0, 1]`` for sRGB unless stated.
* The opponent-color basis is a fixed **orthonormal** 3x3 matrix, so its inverse is
  its transpose and it preserves L2 norm — a clean, documented color space for the
  representation track (decision 0003 ``color_whitened``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# sRGB transfer function constants (IEC 61966-2-1). Standard, not tunable.
_SRGB_THRESHOLD_ENCODED = 0.04045
_SRGB_THRESHOLD_LINEAR = 0.0031308
_SRGB_A = 0.055
_SRGB_GAMMA = 2.4
_SRGB_LINEAR_SLOPE = 12.92


def srgb_to_linear(x: Array) -> Array:
    """Decode gamma-encoded sRGB in ``[0, 1]`` to linear light (IEC 61966-2-1).

    The exact inverse of :func:`linear_to_srgb`. Operates elementwise on any shape.
    """
    x = np.asarray(x, dtype=np.float64)
    low = x <= _SRGB_THRESHOLD_ENCODED
    linear = np.empty_like(x)
    linear[low] = x[low] / _SRGB_LINEAR_SLOPE
    linear[~low] = ((x[~low] + _SRGB_A) / (1.0 + _SRGB_A)) ** _SRGB_GAMMA
    return linear


def linear_to_srgb(x: Array) -> Array:
    """Encode linear-light values in ``[0, 1]`` to gamma-encoded sRGB.

    The exact inverse of :func:`srgb_to_linear`.
    """
    x = np.asarray(x, dtype=np.float64)
    low = x <= _SRGB_THRESHOLD_LINEAR
    encoded = np.empty_like(x)
    encoded[low] = x[low] * _SRGB_LINEAR_SLOPE
    encoded[~low] = (1.0 + _SRGB_A) * np.power(x[~low], 1.0 / _SRGB_GAMMA) - _SRGB_A
    return encoded


# Orthonormal opponent-color basis. Rows are unit-norm and mutually orthogonal:
#   luminance  = (R + G + B) / sqrt(3)
#   red-green  = (R - G)     / sqrt(2)
#   blue-yellow= (R + G - 2B)/ sqrt(6)
# This is a fixed rotation of RGB (not a data-fit), so the inverse is its transpose
# and it preserves the L2 norm of a color vector (decision 0003).
_OPPONENT_MATRIX: Array = np.array(
    [
        [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
        [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0],
        [1.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0)],
    ],
    dtype=np.float64,
)


def opponent_matrix() -> Array:
    """Return a copy of the fixed orthonormal opponent-color matrix ``(3, 3)``."""
    return _OPPONENT_MATRIX.copy()


def rgb_to_opponent(x: Array) -> Array:
    """Rotate channel-last RGB into the orthonormal opponent basis (``(..., 3)``)."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] != 3:
        raise ValueError(f"expected channel-last RGB with last dim 3, got shape {x.shape}")
    return x @ _OPPONENT_MATRIX.T


def opponent_to_rgb(x: Array) -> Array:
    """Inverse of :func:`rgb_to_opponent` (orthonormal, so inverse == transpose)."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] != 3:
        raise ValueError(f"expected channel-last opponent data with last dim 3, got {x.shape}")
    return x @ _OPPONENT_MATRIX  # (M^T)^-1 == M for orthonormal M


def remove_dc(patches: Array) -> tuple[Array, Array]:
    """Remove the per-patch, per-channel mean (DC component).

    ``patches`` has shape ``(..., P, C)`` where ``P`` indexes pixels within a patch
    and ``C`` is channels. Returns ``(centered, dc)`` where ``dc`` (shape
    ``(..., 1, C)``) is the removed mean, so :func:`restore_dc` reverses it exactly.
    """
    patches = np.asarray(patches, dtype=np.float64)
    dc = patches.mean(axis=-2, keepdims=True)
    return patches - dc, dc


def restore_dc(centered: Array, dc: Array) -> Array:
    """Add back a DC component removed by :func:`remove_dc` (exact inverse)."""
    return np.asarray(centered, dtype=np.float64) + np.asarray(dc, dtype=np.float64)
