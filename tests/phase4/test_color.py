"""Tests for color transforms: sRGB, opponent basis, DC removal (Phase 4 task 9)."""

from __future__ import annotations

import numpy as np
import pytest

from neural_repr.data import (
    linear_to_srgb,
    opponent_matrix,
    opponent_to_rgb,
    remove_dc,
    restore_dc,
    rgb_to_opponent,
    srgb_to_linear,
)

RNG = np.random.default_rng(0)


def test_srgb_linear_roundtrip() -> None:
    x = RNG.uniform(0.0, 1.0, (16, 16, 3))
    np.testing.assert_allclose(srgb_to_linear(linear_to_srgb(x)), x, atol=1e-12)
    np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-12)


def test_srgb_endpoints_and_monotone() -> None:
    assert srgb_to_linear(np.array([0.0]))[0] == pytest.approx(0.0)
    assert srgb_to_linear(np.array([1.0]))[0] == pytest.approx(1.0)
    xs = np.linspace(0, 1, 50)
    lin = srgb_to_linear(xs)
    assert np.all(np.diff(lin) > 0)  # strictly increasing


def test_opponent_matrix_is_orthonormal() -> None:
    m = opponent_matrix()
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)


def test_opponent_roundtrip_and_norm_preserving() -> None:
    x = RNG.uniform(0.0, 1.0, (10, 10, 3))
    opp = rgb_to_opponent(x)
    np.testing.assert_allclose(opponent_to_rgb(opp), x, atol=1e-12)
    # Orthonormal => per-pixel L2 norm preserved.
    np.testing.assert_allclose(np.linalg.norm(x, axis=-1), np.linalg.norm(opp, axis=-1), atol=1e-12)


def test_opponent_luminance_channel() -> None:
    # A neutral gray maps all energy to the luminance channel; chroma channels ~0.
    gray = np.full((4, 4, 3), 0.5)
    opp = rgb_to_opponent(gray)
    assert np.allclose(opp[..., 1], 0.0, atol=1e-12)
    assert np.allclose(opp[..., 2], 0.0, atol=1e-12)
    assert np.all(opp[..., 0] > 0)


def test_opponent_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="last dim 3"):
        rgb_to_opponent(np.zeros((4, 4, 4)))


def test_dc_removal_roundtrip_and_zero_mean() -> None:
    patches = RNG.uniform(0.0, 1.0, (5, 256, 3))
    centered, dc = remove_dc(patches)
    np.testing.assert_allclose(centered.mean(axis=-2), 0.0, atol=1e-12)
    np.testing.assert_allclose(restore_dc(centered, dc), patches, atol=1e-12)
    assert dc.shape == (5, 1, 3)
